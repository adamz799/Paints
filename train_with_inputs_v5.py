import cv2, torch, random, time, os
import numpy as np
from torch import nn, optim
from config import *
from tensorboardX import SummaryWriter

from utils_v5 import  get_image, get_tensors, device, tensor_YUV2RGB
from Unet4 import InputNet, UnetD, FeatureNet

writer = SummaryWriter()

BATCH_SIZE = 4
epoch = 200000+1 #480000*4/4
ALPHA = 1e-7
BATE = 10
LAMBDA = 1
ITERATION = 4

file_list = os.listdir(label_dir)

#net = InputNet().to(device)  
netD = UnetD().to(device)
#fnet = FeatureNet().to(device)
net = torch.load('with_cue_v7.pkl').to(device)
#torch.save(lnet,'lnet2.pkl').to(device)
#netD=torch.load( 'netD_v7.pkl').to(device)

L1_loss = nn.L1Loss().to(device)
L2_loss = nn.MSELoss().to(device)
BCE_loss = nn.BCEWithLogitsLoss().to(device)

optimizer = optim.Adam(net.parameters(), lr = 1e-4, betas=(0.9,0.99))
#optimizerL = optim.Adam(lnet.parameters(), lr = 1e-4, betas=(0.9,0.99))
optimizerD = optim.Adam(netD.parameters(), lr = 1e-4, betas=(0.9,0.99))

real_target = torch.ones([BATCH_SIZE, 1, 8, 8]).to(device)
fake_target = torch.zeros(real_target.size()).to(device)


def loss_func_d(i, outputs, label_batch, netD):
    real = netD(label_batch)
    fake = netD(outputs[-1].detach())
    
    d_loss = 0.5*LAMBDA*(L2_loss(real, real_target)+L2_loss(fake, fake_target))
    writer.add_scalar('data1/d_loss', d_loss.item(), i)
    print(' d_loss {:.4f}'.format(d_loss.item()))
    
    return d_loss

def loss_func_g(i, outputs, label_batch, netD):
    # l1loss = 0
    # feature_loss = 0
    # label_feature = fnet(tensor_YUV2RGB(label_batch))
    # for r in outputs:
    #     l1loss += L1_loss(r, label_batch)
    #     feature_loss+=BATE* L1_loss(fnet(tensor_YUV2RGB(r)), label_feature)

    l1loss = L1_loss(outputs[-1], label_batch)
    l1loss_rgb = 8 * L1_loss(tensor_YUV2RGB(outputs[-1]), tensor_YUV2RGB(label_batch))
    #feature_loss=BATE* L1_loss(fnet(tensor_YUV2RGB(outputs[-1])), fnet(tensor_YUV2RGB(label_batch)))

    # l1loss/=BATCH_SIZE
    # feature_loss/=BATCH_SIZE

    g_loss =LAMBDA* 0.5*L2_loss( netD(outputs[-1]), real_target)
    
    loss = l1loss+l1loss_rgb+g_loss #+ ALPHA*l
    # if i>16000:
    #     loss+=g_loss
    
    # if False:#i%5000<1000:
    #     g = outputs.reshape(BATCH_SIZE, 3, -1)
    #     v = torch.var(g,dim=2)
    #     var = ALPHA*-torch.sum(v)
    #     loss += var
    #     writer.add_scalar('data1/var', var.item(), i)
    #     print('content_loss {:6.4f}, feature_loss {:.4f}, var {:.4f}, g_loss {:.4f}'.format(l1loss.item(), feature_loss.item(), var.item(),  g_loss.item()),end=',')
    # else:
    print('content_loss {:6.4f}, l1loss_rgb {:.4f}, g_loss {:.4f}'.format(l1loss.item(),l1loss_rgb.item(), g_loss.item()),end=',')

    writer.add_scalar('data1/content_loss', l1loss.item(), i)
    writer.add_scalar('data1/l1loss_rgb', l1loss_rgb.item(), i)
    writer.add_scalar('data1/g_loss', g_loss.item(), i)
    
    return loss 



gap = 10

if __name__ == '__main__':
    start_time = time.time()
    for i in range(1, epoch+1):
        filenames = random.sample(file_list, BATCH_SIZE)
        names = [f.split('.')[0] for f in filenames]
        inputs_batch, hint_batch, label_batch = get_tensors(names, data_dir, label_dir, with_clue=True)       
        outputs = net(inputs_batch, hint_batch, ITERATION)
        #outputs = net(inputs_batch, inputs_batch/255.0)
        
        
        # input_feature = fnet((label_batch-128.0)/128.0)
        # output_feature = fnet((outputs-128.0)/128.0)
       
        #train 
        loss_g = loss_func_g(i,outputs, label_batch, netD)
        optimizer.zero_grad()
        #optimizerL.zero_grad()
        loss_g.backward()
        optimizer.step()
        #optimizerL.step()

        loss_d = loss_func_d(i,outputs, label_batch, netD)
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
            torch.save(net,'with_cue_v7.pkl')
            #torch.save(lnet,'lnet2.pkl')
            torch.save(netD, 'netD_v7_1.pkl')

    writer.close()
    

    