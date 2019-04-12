import cv2, torch
import numpy as np
from Unet import FeatureNet
from utils_v4 import get_tensors


import time, os

data_dir = 'pics/256_data/' + 'bw_add/'
label_dir = 'pics/256_data/' + 'color_add/'
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# rgb_batch = cv2.imread('D:/Paints/pics/256_data/color_add/1.png')
# rgb_batch = np.transpose(rgb_batch, (2,0,1))
# rgb_batch = rgb_batch[::-1].astype(np.float32)
# for i in range(3):
#     rgb_batch[i]/=255
#     rgb_batch[i]-=mean[i]
#     rgb_batch[i]/=std[i]

net = FeatureNet().eval().cuda()
while 1:

    _, _, rgb_batch = get_tensors(['1'], data_dir, label_dir)  
    output = net(rgb_batch)

print(output)