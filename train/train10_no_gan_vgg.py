from config10 import *
from tensorboardX import SummaryWriter

from utils10 import get_tensors, delta_E1994, tensor_Lab2torch_rgb
from Unet10 import InputNet
from VGG import VGG

writer = SummaryWriter()

BATCH_SIZE = 7
epoch = 100000+1 #1000000/BATCH_SIZE
ALPHA = 1e-7
BATE = 10
LAMBDA = 2
ITERATION = 4
LR = 1e-5

version = 'v10_3_vgg_b'
torch.backends.cudnn.benchmark=True

#net = InputNet().to(device)  
#netD = UnetDL().to(device)
#fnet = FeatureNet().to(device)
net = torch.load('v10_3_vgg.net').to(device)
feature_net = VGG().eval().to(device)

L1_loss = nn.L1Loss().to(device)
L2_loss = nn.MSELoss().to(device)
#BCE_loss = nn.BCEWithLogitsLoss().to(device)

init_lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr = LR, betas=(0.9,0.999))

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

    content_loss = L1_loss(inputs_content, label_content)
    style_loss = L2_loss(inputs_mean, label_mean) + L2_loss(inputs_std, label_std)
    return   style_loss,content_loss

weight = [4/3., 4/3., 2/3., 2/3.]
def loss_func_g(i,outputs, label_batch): 
    l={} 
    color_loss = 0.13*delta_E1994(label_batch, outputs)
    # if i< 4000:
    #     color_loss = color_loss*1.1
    l['color_loss']=color_loss
    content_loss = L1_loss(outputs[:,0], label_batch[:,0])
    l['content_loss']=content_loss
    ab_loss = L1_loss(outputs[:,1:3], label_batch[:,1:3])
    l['ab_loss']=ab_loss
    
    rgb_label = tensor_Lab2torch_rgb(label_batch)
    rgb_outputs = tensor_Lab2torch_rgb(outputs)
    rgb_label = list(feature_net(rgb_label))
    rgb_outputs = list(feature_net(rgb_outputs))

    feature_content_loss = 0
    feature_style_loss = 0
    for j in range(len(rgb_label)):
        style_loss,cont_loss = calc_AdaIN_loss(rgb_label[j],rgb_outputs[j])
        feature_content_loss = feature_content_loss + weight[j]*cont_loss
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
    if i%4==0:
        print(output_str)   
    return total_loss 


gap = 10*4

if __name__ == '__main__':
    start_time = time.time()
    for i in range(1, epoch+1):
        filenames = random.sample(file_list, BATCH_SIZE)
        inputs_batch, hint_batch, label_batch = get_tensors(filenames)       
        outputs = net(inputs_batch, hint_batch, ITERATION)
        
        


        # if i==5000:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = LR
        
        #train 
        loss_g = loss_func_g(i,outputs, label_batch)#, rgb_outputs)       
        loss_g.backward()
        if i%4 == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        
        if i % gap == 0:
            end_time = time.time()
            #print('epoch {}, loss {:.6f}, time {:.3f}s'.format(i,l_1/gap,end_time-start_time))    
            print('epoch {}, time {:.4f}s'.format(i,end_time-start_time))   
            start_time = time.time()
            #l_1=0
            

        if i % 1000 == 0:
            torch.save(net,version+'.net')
            
    writer.close()
    

    