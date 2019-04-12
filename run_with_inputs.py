import cv2, torch, torchvision, random
import numpy as np

from utils import  weight_init, get_image, get_tensors
from Unet import InputNet
from torch import nn, optim
from torchvision import datasets, models, transforms

import time, os

device = torch.device('cuda')

BATCH_SIZE = 1
epoch = 120000

data_dir = 'pics/256_data/' + 'bw_add/'
label_dir = 'pics/256_data/' + 'color_add/'
file_list = os.listdir(label_dir)

net = torch.load('with_cue3_3.pkl').eval()
loss_func = nn.L1Loss().cuda()

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs_batch, label_batch = get_tensors(names, data_dir, label_dir, with_clue=True, data_augm=False)
    outputs = net(inputs_batch)

    loss = loss_func(outputs, label_batch)

    print('epoch {0}, loss {1:.8f}'.format(i,loss.item()))
        
    get_image('input', inputs_batch)
    get_image('label', label_batch)
    get_image('result', outputs)
    cv2.waitKey(0)