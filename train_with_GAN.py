import cv2, torch, torchvision, random, os, time
import numpy as np

from utils import  weight_init, get_image, get_tensors
from Unet import Unet1, UnetD
from torch import nn, optim

device = torch.device('cuda')
#print(torch.get_default_dtype())

BATCH_SIZE = 12
epoch = 120000

line_dir = 'pics/256_data/' + 'bw1/'
color_dir = 'pics/256_data/' + 'color/'
file_list = os.listdir(color_dir)


net = Unet1().cuda()
net.apply(weight_init)
netD = UnetD().cuda()
netD.apply(weight_init)

# net = torch.load('paint_net_G1.pkl')
# netD = torch.load('paint_net_D1.pkl')

loss_func = nn.L1Loss().cuda()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)
optimizerD = optim.Adam(netD.parameters(), lr = 1e-4)

start_time = time.time()


l_1=0
l_2=0
e=0
e_r=0
gap = 10
batch_scale = 1
target_fake = None
target_real = None

for i in range(1, epoch+1):
    filenames = random.sample(file_list, BATCH_SIZE)
    names = [f.split('.')[0] for f in filenames]
    inputs, color_imgs = get_tensors(names, line_dir, color_dir)
    
    outputs = net(inputs)

    #train Discriminator 
    optimizerD.zero_grad()
    output_D_real = netD(color_imgs)
    output_D_fake = netD(outputs.detach())
    if i==1:
        target_real = torch.Tensor(torch.ones(output_D_real.size())).cuda().detach()
        target_fake = torch.Tensor(torch.zeros(output_D_fake.size())).cuda().detach()
    err_D_real = loss_func(output_D_real, target_real)
    err_D_fake = loss_func(output_D_fake, target_fake)
    errD = err_D_fake+err_D_real 
    errD.backward()
    optimizerD.step()
    e+=err_D_fake[0]
    e_r+=err_D_real[0]
   
    #train Generater
    optimizer.zero_grad()
    outputs_D = netD(outputs)
    l1 = loss_func(outputs_D, target_real)#Real/fake loss
    l2 = loss_func(outputs, color_imgs)#content loss
    loss = l1 + l2
    loss.backward()
    optimizer.step()

    l_1+=l1[0]
    l_2+=l2[0]
    if i % gap == 0:
        end_time = time.time()
        print('epoch {0}, G_real_loss {1:.4f}, content_loss {2:.4f}, D_fake_loss {3:.4f}, D_real_loss {4:.4f}, time: {5:.3f}'.format(i,l_1/gap, l_2/gap,e/gap, e_r/gap, end_time-start_time))
        start_time = time.time()
        l_1=0
        l_2=0
        e=0
        e_r=0
    if i % 1000 == 0:
        torch.save(net,'paint_net_G2.pkl')
        torch.save(netD,'paint_net_D2.pkl')
    