import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from utils_v4 import weight_init

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


class InputNet(nn.Module):
    def __init__(self):
        super(InputNet, self).__init__()
        #self.fnet = FeatureNet().eval()   
        self.conv1 = nn.Sequential(
            nn.Conv2d(5,base_channel,3,padding=1),
            # nn.LeakyReLU(0.2,inplace=flag),
            # nn.BatchNorm2d(base_channel),
        )#3-32; 256-256
        self.conv2 = nn.Sequential(           
            nn.Conv2d(base_channel,2*base_channel,4, stride= 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(2*base_channel),  
            nn.Conv2d(2*base_channel,2*base_channel,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(2*base_channel),      
        )#32-64; 256-128
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*base_channel,4*base_channel, 4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(4*base_channel),
            nn.Conv2d(4*base_channel,4*base_channel,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(4*base_channel),        
        )#64-128; 128-64
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*base_channel,8*base_channel,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(8*base_channel),
            nn.Conv2d(8*base_channel,8*base_channel,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(8*base_channel),           
        )#128-256; 64-32
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*base_channel,16*base_channel,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(16*base_channel),
            nn.Conv2d(16*base_channel,16*base_channel,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(16*base_channel),
        )#256-512; 32-16
        self.conv6 = nn.Sequential(
            nn.Conv2d(16*base_channel,32*base_channel,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(32*base_channel)
        )#512-1024; 16-8
        self.dconv6 = nn.Sequential(
            nn.ConvTranspose2d(32*base_channel+2048,16*base_channel, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(16*base_channel),
        )#3072-512; 8-16
        self.dconv5 = nn.Sequential(
            nn.Conv2d(32*base_channel,16*base_channel, 3, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(16*base_channel),
            nn.ConvTranspose2d(16*base_channel,8*base_channel,4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(8*base_channel),
        )#(512+512)->512->256; 16-16-32
        self.dconv4 = nn.Sequential(
            nn.Conv2d(16*base_channel,8*base_channel, 3, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(8*base_channel),
            nn.ConvTranspose2d(8*base_channel,4*base_channel,4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(4*base_channel),
        )#512-128; 32-64
        self.dconv3 = nn.Sequential(
            nn.Conv2d(8*base_channel,4*base_channel, 3, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(4*base_channel),
            nn.ConvTranspose2d(4*base_channel,2*base_channel,4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(2*base_channel),
        )#256-64; 64-128
        self.dconv2 = nn.Sequential(
            nn.Conv2d(4*base_channel,2*base_channel, 3, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(2*base_channel),
            nn.ConvTranspose2d(2*base_channel,base_channel,4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(base_channel),
        )#128-32; 128-256
        self.dconv1 = nn.Sequential(
            nn.Conv2d(2*base_channel,base_channel, 3, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(base_channel),
            nn.Conv2d(base_channel, 3, 3, padding=1),
        )#64-3; 256-256
        
        #init
        for name, modules in self.named_children():
            if name != 'fnet':
                modules.apply(weight_init)

    def forward(self, image, rgb_img):
        self.r1 = self.conv1(image)#1-32; 256-256
        self.r2 = self.conv2(self.r1)#32-64; 256-128
        self.r3 = self.conv3(self.r2)#64-128; 128-64
        self.r4 = self.conv4(self.r3)#128-256; 64-32
        self.r5 = self.conv5(self.r4)#256-512; 32-16
        self.r6 = self.conv6(self.r5)#512-1024; 16-8
        #self.f = self.fnet(rgb_img)#[1,2048,8,8]
        self.d6 = self.dconv6(self.r6)#3072-512
        self.d5 = self.dconv5(torch.cat([self.d6, self.r5],1))#1024-256
        self.d4 = self.dconv4(torch.cat([self.d5, self.r4],1))#512-128
        self.d3 = self.dconv3(torch.cat([self.d4, self.r3],1))#256-64
        self.d2 = self.dconv2(torch.cat([self.d3, self.r2],1))#128-32
        self.d1 = self.dconv1(torch.cat([self.d2, self.r1],1))#64-3
        return self.d1


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
            nn.Conv2d(512,1,3,padding=1)
        )
        #init
        self.apply(weight_init)
        
    def forward(self, image):
        self.r2 = self.conv2(image)#3-64
        self.r3 = self.conv3(self.r2)#64-128
        self.r4 = self.conv4(self.r3)#128-256
        self.r5 = self.conv5(self.r4)#256-512
        return self.r5


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





