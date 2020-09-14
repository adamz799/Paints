from config10 import *
from tensorboardX import SummaryWriter

from utils10 import get_tensors, delta_E1994, tensor_Lab2RGB
from Unet10 import InputNet, UnetD, UnetDL
from VGG import VGG

writer = SummaryWriter()

BATCH_SIZE = 16
epoch = 120000+1 #720000*6/BATCH_SIZE
ALPHA = 1e-7
BATE = 10
LAMBDA = 2
ITERATION = 4
LR = 1e-5

version = 'new_test1'
torch.backends.cudnn.benchmark=True

#net = InputNet().to(device)  
#netD = UnetDL().to(device)
#fnet = FeatureNet().to(device)
net = torch.load('v10_3_2.net').to(device)
netD=torch.load('v10_3_2_D.net').to(device)


L1_loss = nn.L1Loss().to(device)
L2_loss = nn.MSELoss().to(device)
#BCE_loss = nn.BCEWithLogitsLoss().to(device)

init_lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr = LR, betas=(0.9,0.999))
optimizerD = optim.Adam(netD.parameters(), lr = LR, betas=(0.9,0.999))

width = 16
real_target = torch.ones([BATCH_SIZE, 1, width, width]).to(device)
fake_target = torch.zeros(real_target.size()).to(device)

def loss_func_d(i,outputs, label_batch):
    real = netD(label_batch)
    fake = netD(outputs.detach())   
    d_loss = 0.5*LAMBDA*(L2_loss(real, real_target)+L2_loss(fake, fake_target))
    v = d_loss.item()
    writer.add_scalar(version+'/d_loss', v, i)
    print('d_loss {:.4f}'.format(v))   
    return d_loss

def loss_func_g(i,outputs, label_batch): 
    l={} 
    color_loss = 0.1*delta_E1994(label_batch, outputs)
    l['color_loss']=color_loss
    content_loss = L1_loss(outputs[:,0], label_batch[:,0])
    l['content_loss']=content_loss
    ab_loss = L1_loss(outputs[:,1:3], label_batch[:,1:3])
    l['ab_loss']=ab_loss
    #if i>10000:
    g_loss =LAMBDA* 0.5*L2_loss( netD(outputs), real_target)
    l['g_loss']=g_loss
    output_str = ''
    total_loss=0
    for loss_name, loss_val in l.items():
        v = loss_val.item()
        output_str+=loss_name+' {:.4f}, '.format(v)
        writer.add_scalar(version+'/'+loss_name, v, i)
        total_loss+=loss_val
    print(output_str, end='')   
    return total_loss 


gap = 10

if __name__ == '__main__':
    start_time = time.time()
    for i in range(1, epoch+1):
        filenames = random.sample(file_list, BATCH_SIZE)
        inputs_batch, hint_batch, label_batch = get_tensors(filenames)       
        outputs = net(inputs_batch, hint_batch, ITERATION)
        
        # rgb_label = tensor_Lab2RGB(label_batch)
        # rgb_outputs = tensor_Lab2RGB(outputs)
        # if i==60000:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.5*LR
        #     for param_group in optimizerD.param_groups:
        #         param_group['lr'] = 0.5*LR
                
        #train 
        loss_g = loss_func_g(i,outputs, label_batch)#, rgb_outputs)
        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()
        #optimizerL.step()
        # loss_d = loss_func_d(i,outputs, label_batch)#, rgb_outputs, rgb_label)
        # optimizerD.zero_grad()
        # loss_d.backward()
        # optimizerD.step()
        if i%10>4:
            loss_d = loss_func_d(i,outputs, label_batch)#, rgb_outputs, rgb_label)
            optimizerD.zero_grad()
            loss_d.backward()
            optimizerD.step()
        else:
            print('n')
        
        #if i % batch_scale == 0:
        
        #l_1+=(loss_g.item()+loss_d.item())
        
        if i % gap == 0:
            end_time = time.time()
            #print('epoch {}, loss {:.6f}, time {:.3f}s'.format(i,l_1/gap,end_time-start_time))    
            print('epoch {}, time {:.4f}s'.format(i,end_time-start_time))   
            start_time = time.time()
            #l_1=0
            

        if i % 1000 == 0:
            torch.save(net,version+'.net')
            torch.save(netD, version+'_D.net')

    writer.close()
    

    