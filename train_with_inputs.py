import cv2, torch, torchvision, random,  concurrent.futures
import numpy as np

from utils import  weight_init, get_image, get_tensors
from Unet import InputNet, UnetD, FeatureNet
from torch import nn, optim
from torchvision import datasets, models, transforms

from tensorboardX import SummaryWriter

import time, os

device = torch.device('cuda')
writer = SummaryWriter()

BATCH_SIZE = 8
epoch = 100000+1 #400000*4*1.15/16=115000
ALPHA = 1e-7
LAMBDA = 1
TTT =0 

data_dir = 'pics/256_data/' + 'bw_add/'
label_dir = 'pics/256_data/' + 'color_add/'
file_list = os.listdir(label_dir)

# net = InputNet().cuda()   
# net.apply(weight_init)
fnet = FeatureNet().cuda() 
fnet.net.weight.requires_grad = False
# netD = UnetD().cuda()
# netD.apply(weight_init)

net = torch.load('with_cue3_2.pkl')
netD = torch.load('netD3_2.pkl')

L1_loss = nn.L1Loss().cuda()
L2_loss = nn.MSELoss().cuda()

optimizer = optim.Adam(net.parameters(), lr = 1e-4, betas=(0.9,0.99))
#optimizerL = optim.Adam(lnet.parameters(), lr = 1e-4, betas=(0.9,0.99))
optimizerD = optim.Adam(netD.parameters(), lr = 1e-4, betas=(0.9,0.99))

real_target = torch.ones([BATCH_SIZE, 3, 16, 16]).cuda()
fake_target = torch.zeros(real_target.size()).cuda()


def loss_func_d(outputs, label_batch, netD):
    real = netD(label_batch)
    fake = netD(outputs.detach())
    
    # Setting target close to 0 and 1 instead the exact value to protect the loss from being inf
    # d1 = BCE_loss(real, torch.ones(real.size()).cuda()*0.9)
    # d2 = BCE_loss(fake, torch.ones(fake.size()).cuda()*0.1)
    d_loss = 0.5*LAMBDA*(L2_loss(real, real_target)+L2_loss(fake, fake_target))
    writer.add_scalar('data1/d_loss', d_loss.item(), TTT)
    print(' d_loss {:.4f}'.format(d_loss.item()))
    
    return d_loss

def loss_func_g(outputs, label_batch, netD, l_input, l_output):
    l1loss = L1_loss(outputs, label_batch)
    l_l1loss = L1_loss(l_input, l_output)
    g = outputs.reshape(BATCH_SIZE, 3, -1)
    v = torch.var(g,dim=2)
    l = -torch.sum(v)
    
    g_loss =LAMBDA* 0.5*L2_loss( netD(outputs), real_target)
    
    global TTT
    writer.add_scalar('data1/content_loss', l1loss.item(), TTT)
    writer.add_scalar('data1/l_content_loss', l_l1loss.item(), TTT)
    writer.add_scalar('data1/L', l.item(), TTT)
    writer.add_scalar('data1/g_loss', g_loss.item(), TTT)
    TTT+=1
    print('l1 {:2.4f}, fnet {:.4f}, var {:.4f}, g_loss {:.4f}'.format(l1loss.item(), 10*l_l1loss.item(), l.item(),  g_loss.item()),end=',')
    
    loss = l1loss +10 * l_l1loss + g_loss #+ ALPHA*l
    # if TTT>20000:
    #     loss+=g_loss
    return loss 

start_time = time.time()

l_1=0
gap = 10
batch_scale = 2

if __name__ == '__main__':

    for i in range(1, epoch+1):
        filenames = random.sample(file_list, BATCH_SIZE)
        names = [f.split('.')[0] for f in filenames]
        inputs_batch, label_batch = get_tensors(names, data_dir, label_dir, with_clue=True)       
        outputs = net(inputs_batch)

        l_input = fnet((label_batch-128.0)/128.0)
        l_output = fnet((outputs-128.0)/128.0)
       
        #train 
        loss_g = loss_func_g(outputs, label_batch, netD, l_input, l_output)
        optimizer.zero_grad()
        #optimizerL.zero_grad()
        loss_g.backward()
        optimizer.step()
        #optimizerL.step()

        loss_d = loss_func_d(outputs, label_batch, netD)
        optimizerD.zero_grad()
        loss_d.backward()
        optimizerD.step()
        
        #if i % batch_scale == 0:
        
        #l_1+=(loss_g.item()+loss_d.item())
        
        if i % gap == 0:
            end_time = time.time()
            #print('epoch {}, loss {:.6f}, time {:.3f}s'.format(i,l_1/gap,end_time-start_time))    
            print('epoch {}, time {:.4f}s'.format(i,end_time-start_time))   
            start_time = time.time()
            #l_1=0
            

        if i % 1000 == 0:
            torch.save(net,'with_cue3_3.pkl')
            #torch.save(lnet,'lnet2.pkl')
            torch.save(netD, 'netD3_3.pkl')

    writer.close()
    

    