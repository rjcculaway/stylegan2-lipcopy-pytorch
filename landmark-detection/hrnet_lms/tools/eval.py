# ------------------------------------------------------------------------------
# Created by Gaofeng(lfxx1994@gmail.com)
# ------------------------------------------------------------------------------

import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from PIL import Image
import numpy as np
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds
import matplotlib; matplotlib.use('agg')


from matplotlib import pyplot as plt


# python tools/eval.py --imagepath /home/code-base/user_space/Data/FFHQ/images1024x1024/00000.png

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg',
                        default='experiments/300w/face_alignment_300w_hrnet_w18.yaml',
                        help='experiment configuration filename', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='./hrnetv2_pretrained/hrnet-300w.pth', type=str)
    parser.add_argument('--imagepath', help='Path of the image to be detected', default='111.jpg',
                        type=str)
    parser.add_argument('--face', nargs='+', type=float, default=[1024//8, 1024//8, 1024-1024//8, 1024-1024//8],
                        help='The coordinate [x1,y1,x2,y2] of a face')
    args = parser.parse_args()
    update_config(config, args)
    return args


def prepare_input(image, bbox, image_size):
    """
    :param image:The path to the image to be detected
    :param bbox:The bbox of target face
    :param image_size: refers to config file
    :return:
    """
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    center_w = (bbox[0] + bbox[2]) / 2
    center_h = (bbox[1] + bbox[3]) / 2
    center = torch.Tensor([center_w, center_h])
    scale *= 1.25
    img = np.array(Image.open(image).convert('RGB'), dtype=np.float32)
    img_np = img.copy().astype(np.uint8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = crop(img, center, scale, image_size, rot=0)
    print(f'>>>> cropped image shape and bbox: {img.shape} and {bbox}')
    img = img.astype(np.float32)
    img = (img / 255.0 - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    return img, center, scale, img_np


def main():
    args = parse_args()
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)
    if config.GPUS is list:
        gpus = list(config.GPUS)
    else:
        gpus = [config.GPUS]
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    model.load_state_dict(state_dict)
    model.eval()
    inp, center, scale, img_np = prepare_input(args.imagepath, args.face, config.MODEL.IMAGE_SIZE)
    output = model(inp)
    score_map = output.data.cpu()
    import pdb; pdb.set_trace()
    preds = decode_preds(score_map, center, scale, [64, 64])
    preds = preds.numpy()
    # cv2.namedWindow('test', 0)
    img_once = cv2.imread(args.imagepath)
    for i in preds[0, :, :]:
        cv2.circle(img_once, tuple(list(int(p) for p in i.tolist())), 2, (255, 255, 0), 1)
    
    print('>>>> saving ...')
    
    # import pdb; pdb.set_trace()
    plt.imsave('lms.png', img_once[:,:,::-1])
    plt.imsave('original.png', img_np)
    plt.imsave('results.png', np.concatenate([img_np, img_once[:,:,::-1]],axis=1))

    
if __name__ == '__main__':
    main()
