import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

flag = True

#torch.set_default_tensor_type(torch.Float())

class Unet1(nn.Module):
    def __init__(self):
        super(Unet1, self).__init__()
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
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(256,512,4, stride= 2,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(512),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(512),
        )
        self.dconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024,512, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,256,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(256),
            #nn.Upsample(scale_factor= 2),
        )
        self.dconv4 = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,128,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(128),
            #nn.Upsample(scale_factor= 2)
        )
        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(256,128, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),
        )
        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride = 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,3,3,padding=1),
            # nn.ReLU(inplace=flag),
            # nn.BatchNorm2d(32),           
        )
        #self.dconv1 = nn.Conv2d(64,3,1)#BGR-image

    def forward(self, image):
        #self.r1 = self.conv1(image)#1-32
        self.r2 = self.conv2(image)#3-64; 256-128
        self.r3 = self.conv3(self.r2)#64-128; 128-64
        self.r4 = self.conv4(self.r3)#128-256; 64-32
        self.r5_1 = self.conv5_1(self.r4)#256-512; 32-16
        self.r5_2 = self.conv5_2(self.r5_1)#512-512
        #self.dr5 = F.interpolate(self.r5, size=(self.r4.size()[2:4]))
        self.d5 = self.dconv5(torch.cat([self.r5_1, self.r5_2],1))#(n,channel,height,width) 1024-256
        #self.dr4 = F.interpolate(self.temp4, size=self.r3.size()[2:4])
        self.d4 = self.dconv4(torch.cat([self.r4,self.d5],1))#512-128
        self.d3 = self.dconv3(torch.cat([self.r3,self.d4],1))#256-64
        self.d2 = self.dconv2(torch.cat([self.r2,self.d3],1))#128-32
        #self.d1 = self.dconv1(torch.cat([self.r1,self.d2],1))#64-3
        return self.d2


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
        
    def forward(self, image):
        self.r2 = self.conv2(image)#3-64
        self.r3 = self.conv3(self.r2)#64-128
        self.r4 = self.conv4(self.r3)#128-256
        self.r5 = self.conv5(self.r4)#256-512
        return self.r5


class InputNet(Unet1):
    def __init__(self):
        super(InputNet, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(5,64,4, stride= 2, padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),
            nn.LeakyReLU(0.2,inplace=flag),
            nn.BatchNorm2d(64),
       )



model = models.resnet50(pretrained=True)
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.net = nn.Sequential(*list(model.children())[:-2])

    def forward(self, inputs):
        return self.net(inputs)

