import cv2, torch, random
import numpy as np

from utils_v5 import  get_image, get_tensors, device, tensor_YUV2RGB
from torch import nn, optim

import time, os


BATCH_SIZE = 1
epoch = 120000

data_dir = 'pics/256_data/' + 'bw_add/'
label_dir = 'pics/256_data/' + 'color_add/'
file_list = os.listdir(label_dir)

net = torch.load('with_cue_v7.pkl').to(device)
net1 = torch.load('with_cue3_3.pkl').to(device)
loss_func = nn.L1Loss().cuda()

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs_batch, hint_batch, label_batch = get_tensors(names, data_dir, label_dir, with_clue=True, data_augm=False)
    outputs = net(inputs_batch, hint_batch, 4)
    outputs1 = net1(torch.cat([inputs_batch, hint_batch], 1))
    
        
    get_image('input', hint_batch)
    get_image('label', label_batch)
    for i in range(4):
        get_image('result {}'.format(i), outputs[i])
    get_image('result1', outputs1)

    label_batch = tensor_YUV2RGB(label_batch)*255
    o = tensor_YUV2RGB(outputs[-1])*255
    o1 = tensor_YUV2RGB(outputs1)*255
    
    loss = loss_func(o, label_batch)
    loss1 = loss_func(o1, label_batch)

    print('epoch {0}, loss {1:.6f}, loss1 {2:.6f}'.format(i,loss.item(),loss1.item()))
    cv2.waitKey(0)