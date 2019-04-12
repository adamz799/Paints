import cv2, torch, random
import torch.utils.data as data
#import numpy as np

from utils import  weight_init, get_tensors
from Unet import Unet1
from torch import nn, optim
from paint_data_loader import PaintDataset

import time, os

device = torch.device('cuda')

BATCH_SIZE = 16
epoch = 200000

line_dir = 'pics/256_data/' + 'bw1/'
color_dir = 'pics/256_data/' + 'color/'
file_list = os.listdir(color_dir)

# dataset = PaintDataset('pics/256_data/')
# r_sampler = data.RandomSampler(dataset)
# batch_sampler = data.BatchSampler(r_sampler, BATCH_SIZE, drop_last = False)
# loader = data.DataLoader(dataset, batch_sampler= batch_sampler, num_workers=3)

# net = Unet1().cuda()
# net.apply(weight_init)
net = torch.load('paint_net_new_n.pkl')
#loss_func = nn.MSELoss()
loss_func = nn.L1Loss().cuda()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)
l=0
gap = 10
batch_scale = 2
start_time = time.time()

#if __name__ == '__main__':
for i in range(1, epoch+1):
    
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs, color_imgs = get_tensors(names, line_dir, color_dir)
# for i, data in enumerate(loader, 0):
#     inputs = data[0].cuda()
#     color_imgs = data[1].cuda()
#inputs, color_imgs = list(loader)

    outputs = net(inputs)
    #loss = (outputs-label_batch).pow(2).sum()
    loss = loss_func(outputs, color_imgs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss[0]
    if i % gap == 0:
        end_time = time.time()
        print('epoch {}, loss {}, time {}'.format(i,loss[0],end_time-start_time))
        l=0
        start_time = time.time()
    if i % 1000 == 0:
        torch.save(net,'paint_net_new_n.pkl')