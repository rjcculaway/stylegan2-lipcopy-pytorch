import os
import argparse
import sys


import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hrnet_lms'))
import hrnet_lms.hrnet_lib.models as models
from hrnet_lms.hrnet_lib.config import config, update_config


class options():
    cfg = './hrnet_lms/experiments/300w/face_alignment_300w_hrnet_w18.yaml'
    
    face = [1024//8, 1024//8, 1024-1024//8, 1024-1024//8]

    
class HRNet_lms(nn.Module):
    
    def __init__(self):
        super(HRNet_lms, self).__init__()
        args = options()
        args = update_config(config, args)
        self.model_file = './pretrained_models/hrnet-300w.pth'
        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        
        self.model = models.get_face_alignment_net(config)
        if config.GPUS is list:
            gpus = list(config.GPUS)
        else:
            gpus = [config.GPUS]
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()
        self.model.load_state_dict(torch.load(self.model_file))

        ### unpack DataParallel
        self.model = self.model.module
        self.model.eval()
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, x):
        # input x
        # size : [256,256]
        # data range [-1,1]
        # RGB 
        # output lms coords
        # size: [bs, lms, xy]
        
        # normalize 
        b, c, h, w= x.size()
        x = ((x+1)/2 - self.mean)/self.std
        x = x[:,[2, 1, 0],:,:]
        
        # model forward
        heatmap = self.model(x)
        return heatmap
    
    @staticmethod
    def top_k_indices(x, k):
        """Returns the k largest element indices from a numpy array. You can find
        the original code here: https://stackoverflow.com/q/6910641
        """
        flat = x.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, x.shape)

    def get_peak_location(self, heatmap, image_size=(256, 256)):
        """Return the interpreted location of the top 2 predictions."""
        h_height, h_width = heatmap.shape
        [y1, y2], [x1, x2] = self.top_k_indices(heatmap, 2)
        x = (x1 + (x2 - x1)/4) / h_width * image_size[0]
        y = (y1 + (y2 - y1)/4) / h_height * image_size[1]
        
        return [int(x), int(y)]


    def parse_heatmaps(self, heatmaps, image_size):
        # Parse the heatmaps to get mark locations.
        marks = []
        for heatmap in heatmaps:
            marks.append(self.get_peak_location(heatmap, image_size))

        # Show individual heatmaps stacked.
        heatmap_grid = np.hstack(heatmaps[:8])
        for row in range(1, 12, 1):
            heatmap_grid = np.vstack(
                [heatmap_grid, np.hstack(heatmaps[row:row+8])])

        return np.array(marks), heatmap_grid


    @staticmethod
    def draw_marks(image, marks):
        for m in marks:
            for mark in m:
                # print(mark)
                cv2.circle(image, (int(mark[0]), int(mark[1])), 2, (255, 0, 0), -1)

                
                
if __name__ == '__main__':
    from landmark_detection import HRNet_lms
    
    from PIL import Image
    model = HRNet_lms()
    
    img = np.array(Image.open('/home/code-base/user_space/__Data_old/FFHQ/images1024x1024/00000.png').convert('RGB'), dtype=np.float32) /255.
    img = img * 2 - 1
    print(img.min(), img.max())
    
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    pool = nn.AvgPool2d((4,4))
    x = pool(x)
    print(x.size())
    with torch.no_grad():
        heatmaps = model(x)
        heatmaps = heatmaps.cpu().numpy()
        
        print(heatmaps.shape)
    
    mark_group = []
    for heatmap in heatmaps:
        print(heatmap.shape)
        marks, heatmap_grid = model.parse_heatmaps(heatmap, (256,256))
        mark_group.append(marks)
    
    np_img = (img + 1) / 2 * 255
    np_img = cv2.resize(np_img, (256,256), interpolation = cv2.INTER_AREA)

    model.draw_marks(np_img, mark_group)    
    cv2.imwrite('results.png', np_img[:,:,::-1])
    cv2.imwrite('heatmap.png', heatmap_grid*255)
    
    
    