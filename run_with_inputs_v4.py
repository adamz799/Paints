import cv2, torch, random
import numpy as np

from utils_v4 import  get_image, get_tensors, device
from torch import nn, optim

import time, os


BATCH_SIZE = 1
epoch = 120000

data_dir = 'pics/256_data/' + 'bw_add/'
label_dir = 'pics/256_data/' + 'color_add/'
file_list = os.listdir(label_dir)

net = torch.load('with_cue_v6_1.pkl').to(device)
net1 = torch.load('with_cue3_3.pkl').to(device)
loss_func = nn.L1Loss().cuda()

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs_batch, label_batch = get_tensors(names, data_dir, label_dir, with_clue=True, data_augm=False)
    outputs = net(inputs_batch)
    outputs1 = net1(inputs_batch)
    loss = loss_func(outputs, label_batch)
    loss1 = loss_func(outputs1, label_batch)

    print('epoch {0}, loss {1:.6f}, loss1 {2:.6f}'.format(i,loss.item(),loss1.item()))
        
    get_image('input', inputs_batch)
    get_image('label', label_batch)
    get_image('result', outputs)
    get_image('result1', outputs1)
    cv2.waitKey(0)