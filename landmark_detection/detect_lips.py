import sys
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn

from landmark_detection import HRNet_lms
from PIL import Image

if __name__ == '__main__':
  model = HRNet_lms()

  paths = glob.glob("../sample/*.png")

  for i, path in enumerate(paths):

    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    img = img * 2 - 1 # Map values to 0.0 to 1.0
    
    x = torch.from_numpy(img).to("cuda:0").permute(2, 0, 1) # RGB -> BRG
    x = x.unsqueeze(0)  # Add an additional dimension
    pool = nn.AvgPool2d(4, 4)
    x = pool(x) # Downsample to 25% by average pooling

    with torch.no_grad():
      heatmaps = model(x).cpu().numpy()
      print(heatmaps.shape)

    mark_group = []

    for heatmap in heatmaps:
      print(heatmap.shape)
      marks, heatmap_grid = model.parse_heatmaps(heatmap, (256, 256))
      mark_group.append(marks)

    np_img = (img + 1) / 2 * 255 # -> Map image back to 0-255
    np_img = cv2.resize(np_img, (256, 256), interpolation = cv2.INTER_AREA)

    model.draw_marks(np_img, mark_group)
    cv2.imwrite('results' + str(i + 1) + '.png', np_img[:,:,::-1])
    cv2.imwrite('heatmap' + str(i + 1) + '.png', heatmap_grid*255)
  

