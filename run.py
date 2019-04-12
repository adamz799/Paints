import cv2, torch, torchvision, random
import numpy as np

from utils import  weight_init
from Unet import Unet1
from torch import nn, optim
from torchvision import datasets, models, transforms

import time, os

device = torch.device('cuda')
#print(torch.get_default_dtype())

def get_bw_tensor(filename):
    img = cv2.imread(filename ,cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0
    img-=0.5
    return torch.unsqueeze(torch.from_numpy(img),0)

def get_tensor(filename):
    img = cv2.imread(filename)
    img = np.transpose(img, (2,0,1)).astype(np.float32)/255.0
    img-=0.5
    return torch.from_numpy(img)


def get_image(name, tensor):
    img = torch.squeeze(tensor.cpu().detach(),0).numpy()
    max_v, min_v = img.max(),img.min()
    img+=0.5
    # img = np.where(img<0,0,img)
    # img = np.where(img>1,1,img)
    #img = (img -min_v)/(max_v-min_v)
    img = np.transpose(img*255.0,(1,2,0)).astype(np.uint8)
    cv2.imshow(name,img)
    

BATCH_SIZE = 1
epoch = 10000

data_dir = 'pics/256_data/' + 'bw1/'
label_dir = 'pics/256_data/' + 'color/'
file_list = os.listdir(label_dir)


# net = Unet1().cuda(device)
# net.apply(weight_init)
net = torch.load('paint_net_new_n.pkl')
loss_func = nn.L1Loss().cuda()
#optimizer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay=1e-5)
l=0
gap = 10
batch_scale = 2
for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs_batch = torch.stack([get_bw_tensor(data_dir+f+'.png') for f in names]).to(device)
    label_batch = torch.stack([get_tensor(label_dir+f+'.jpg') for f in names]).to(device)
    # if inputs_batch.size() != label_batch.size():
    #     continue
    # inputs_batch =  torch.stack([get_tensor('709853.jpg')]).to(device)
    # label_batch = inputs_batch
    outputs = net(inputs_batch)
    get_image('o', inputs_batch)
    get_image('l', label_batch)
    get_image('r', outputs)
    
    #loss = (outputs-label_batch).pow(2).sum()
    loss = loss_func(outputs, label_batch)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #l=
    
    print('epoch {}, loss {:6f}'.format(i,loss.item()))
    cv2.waitKey(0)
