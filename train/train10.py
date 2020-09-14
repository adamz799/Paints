from config10 import *
from tensorboardX import SummaryWriter

from utils10 import get_tensors, delta_E1994, tensor_Lab2RGB, tensor_Lab2torch_rgb
#from Unet10 import InputNet, UnetD
from VGG import VGG

writer = SummaryWriter()

BATCH_SIZE = 6
epoch = 100000+1 #1000000*6/BATCH_SIZE
ALPHA = 1e-7
BATE = 10
LAMBDA = 2
ITERATION = 4
LR = 1e-5

version = 'v10_vgg_gan'
torch.backends.cudnn.benchmark=True

#net = InputNet().to(device)  
#netD = UnetDL().to(device)
#fnet = FeatureNet().to(device)
net = torch.load('v10_vgg_gan.net').to(device)
netD=torch.load('v10_vgg_gan_D.net').to(device)
feature_net = VGG().eval().to(device)


L1_loss = nn.L1Loss().to(device)
L2_loss = nn.MSELoss().to(device)
#BCE_loss = nn.BCEWithLogitsLoss().to(device)

init_lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr = init_lr, betas=(0.9,0.999))
optimizerD = optim.Adam(netD.parameters(), lr = LR, betas=(0.9,0.999))

width = 16
real_target = torch.ones([BATCH_SIZE, 1, width, width]).to(device)
fake_target = torch.zeros(real_target.size()).to(device)

def calc_AdaIN_loss(inputs, label):
    shape = inputs.shape
    inputs = inputs.view(shape[0],shape[1],-1)
    label = label.view(shape[0],shape[1],-1)

    inputs_mean = torch.mean(inputs,  dim=2, keepdim=True)
    inputs_std = torch.std(inputs, dim=2, keepdim=True)+1e-7
    label_mean = torch.mean(label,  dim=2, keepdim=True)
    label_std = torch.std(label,  dim=2, keepdim=True)+1e-7

    inputs_content = (inputs-inputs_mean)/inputs_std
    label_content = (label-label_mean)/label_std

    content_loss = L2_loss(inputs_content, label_content)
    style_loss = L2_loss(inputs_mean, label_mean) + L2_loss(inputs_std, label_std)
    return   style_loss,content_loss

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
    color_loss = 0.12*delta_E1994(label_batch, outputs)
    l['color_loss']=color_loss
    content_loss = L1_loss(outputs[:,0], label_batch[:,0])
    l['content_loss']=content_loss
    ab_loss = L1_loss(outputs[:,1:3], label_batch[:,1:3])
    l['ab_loss']=ab_loss
    #if i>10000:
    g_loss =LAMBDA* 0.5*L2_loss( netD(outputs), real_target)
    l['g_loss']=g_loss

    rgb_label = tensor_Lab2torch_rgb(label_batch)
    rgb_outputs = tensor_Lab2torch_rgb(outputs)
    rgb_label = list(feature_net(rgb_label))
    rgb_outputs = list(feature_net(rgb_outputs))

    feature_content_loss = 0
    feature_style_loss = 0
    for j in range(len(rgb_label)):
        style_loss,cont_loss = calc_AdaIN_loss(rgb_label[j],rgb_outputs[j])
        feature_content_loss = feature_content_loss + cont_loss
        feature_style_loss = feature_style_loss +style_loss
    l['feature_content_loss']=feature_content_loss
    l['feature_style_loss']=0.2*feature_style_loss


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
        if i==5000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        #     for param_group in optimizerD.param_groups:
        #         param_group['lr'] = 0.5*LR
                
        #train 
        loss_g = loss_func_g(i,outputs, label_batch)#, rgb_outputs)
        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()
        #optimizerL.step()
        loss_d = loss_func_d(i,outputs, label_batch)#, rgb_outputs, rgb_label)
        optimizerD.zero_grad()
        loss_d.backward()
        optimizerD.step()
        # if i%10>4:
        #     loss_d = loss_func_d(i,outputs, label_batch)#, rgb_outputs, rgb_label)
        #     optimizerD.zero_grad()
        #     loss_d.backward()
        #     optimizerD.step()
        # else:
        #     print('n')
        
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
    

    