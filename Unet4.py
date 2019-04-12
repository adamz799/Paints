import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from utils_v4 import weight_init, device

flag = True

#torch.set_default_tensor_type(torch.Float())
base_channel = 32

model = models.resnet50(pretrained=True)
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.net = nn.Sequential(*list(model.children())[:-2])
    def forward(self, inputs):
        return self.net(inputs)

def squeeze_block(base_channel):
    return nn.Sequential(           
        nn.Conv2d(base_channel,2*base_channel,4, stride= 2, padding=1),
        nn.LeakyReLU(0.2,inplace=flag),
        nn.BatchNorm2d(2*base_channel),  
        nn.Conv2d(2*base_channel,2*base_channel,3,padding=1),
        nn.LeakyReLU(0.2,inplace=flag),
        nn.BatchNorm2d(2*base_channel),      
    )

def unsqueeze_block(base_channel):
    return nn.Sequential(
        nn.Conv2d(4*base_channel,2*base_channel, 3, padding=1),
        nn.LeakyReLU(0.2,inplace=flag),
        nn.BatchNorm2d(2*base_channel),
        nn.ConvTranspose2d(2*base_channel,base_channel,4, stride = 2, padding=1),
        nn.LeakyReLU(0.2,inplace=flag),
        nn.BatchNorm2d(base_channel),
    )

def gate(base_channel):
    return nn.Sequential(
        nn.Conv2d(2*base_channel,base_channel,3,padding=1),
        nn.Sigmoid()
    )



class InputNet(nn.Module):
    def __init__(self):
        super(InputNet, self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(5,base_channel,3,padding=1),
            # nn.LeakyReLU(0.2,inplace=flag),
            # nn.BatchNorm2d(base_channel),
        )#3-32; 256-256
        self.conv2 = squeeze_block(base_channel)#32-64; 256-128
        self.conv3 = squeeze_block(2*base_channel)#64-128; 128-64
        self.conv4 = squeeze_block(4*base_channel)#128-256; 64-32
        self.conv5 = squeeze_block(8*base_channel)#256-512; 32-16
        # self.conv5_2 = nn.Sequential(
        #     nn.Conv2d(16*base_channel,16*base_channel,3,padding=1),
        #     nn.LeakyReLU(0.2,inplace=flag),
        #     nn.BatchNorm2d(16*base_channel),
        # )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16*base_channel,32*base_channel,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(32*base_channel)
        )#512-1024; 16-8
        self.dconv6 = nn.Sequential(
            nn.ConvTranspose2d(32*base_channel,16*base_channel, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(16*base_channel),
        )#3072-512; 8-16
        self.dconv5 = unsqueeze_block(8*base_channel)#(512+512)->512->256; 16-16-32
        self.dconv4 = unsqueeze_block(4*base_channel)#512-128; 32-64
        self.dconv3 = unsqueeze_block(2*base_channel)#256-64; 64-128
        self.dconv2 = unsqueeze_block(base_channel)#128-32; 128-256
        self.dconv1 = nn.Sequential(
            nn.Conv2d(base_channel,3, 3, padding=1),
            # nn.LeakyReLU(0.2,inplace=flag),
            # nn.BatchNorm2d(base_channel),
            # nn.Conv2d(base_channel, 3, 3, padding=1),
        )#32-3; 256-256
        
        #LSTM part
        self.conv_i = gate(32*base_channel)#input gate
        self.conv_f = gate(32*base_channel)#forget gate
        self.conv_o = gate(32*base_channel)#output gate
        self.conv_c_ = nn.Sequential(
            nn.Conv2d(64*base_channel,32*base_channel,3,padding=1),
            nn.Tanh()
        )

        #init
        base_net_dict = torch.load('with_cue_v6_2.pkl').state_dict()
        self_dict = self.state_dict()
        base_net_dict = {name:weight for name,weight in base_net_dict.items() if name in self_dict}
        self_dict.update(base_net_dict)
        self.load_state_dict(self_dict)
        for name, modules in self.named_children():
            if name[0:5]=='conv_':
                modules.apply(weight_init)


    def forward(self, gray_image, hint, iteration):
        output_img = []
        c = None
        h = None
        for i in range(iteration):
            image = torch.cat([gray_image, hint],1)#2+3

            self.r1 = self.conv1(image)#5-32; 256-256
            self.r2 = self.conv2(self.r1)#32-64; 256-128
            self.r3 = self.conv3(self.r2)#64-128; 128-64
            self.r4 = self.conv4(self.r3)#128-256; 64-32
            self.r5 = self.conv5(self.r4)#256-512; 32-16
            self.r6 = self.conv6(self.r5)#512-1024; 16-8
            
            if i==0:
                c = torch.zeros(self.r6.shape).to(device)
                h = torch.zeros(self.r6.shape).to(device)
            
            x = torch.cat([self.r6, h],1)
            f = self.conv_f(x)
            i = self.conv_i(x)
            c_ = self.conv_c_(x)
            o = self.conv_o(x)
            c = f*c + i*c_
            h = o*F.tanh(c)

            #self.f = self.fnet(rgb_img)#[1,2048,8,8]
            self.d6 = self.dconv6(h)#1024-512
            self.d5 = self.dconv5(torch.cat([self.d6, self.r5],1))#1024-256
            self.d4 = self.dconv4(torch.cat([self.d5, self.r4],1))#512-128
            self.d3 = self.dconv3(torch.cat([self.d4, self.r3],1))#256-64
            self.d2 = self.dconv2(torch.cat([self.d3, self.r2],1))#128-32
            self.d1 = self.dconv1(self.d2)#32-3

            hint = self.d1
            output_img.append(self.d1)

        return output_img


class UnetD(nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(3,64,4, stride= 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),         
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, 4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(128),       
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(256),         
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256,512,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,padding=1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512,1024,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,1,3,padding=1)
        )
        #init
        self.apply(weight_init)
        
    def forward(self, image):
        self.r2 = self.conv2(image)#3-64
        self.r3 = self.conv3(self.r2)#64-128
        self.r4 = self.conv4(self.r3)#128-256
        self.r5 = self.conv5(self.r4)#256-512
        self.r6 = self.conv6(self.r5)#512-1024
        return self.r6


# class InputNet(Unet1):
#     def __init__(self):
#         super(InputNet, self).__init__()
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(6,64,4, stride= 2, padding=1),
#             nn.LeakyReLU(0.2,inplace=flag),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64,64,3,padding=1),
#             nn.LeakyReLU(0.2,inplace=flag),
#             nn.BatchNorm2d(64),
#        )





