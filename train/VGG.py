import torch
from torch import nn
from torchvision import models
from config import device

def fix(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.requires_grad = False
    

#model = models.vgg19(pretrained=True)
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        layers = list(models.vgg19(pretrained=True).features.children())
        self.relu1_1 = nn.Sequential(*layers[:2])
        self.relu2_1 = nn.Sequential(*layers[2:7])
        self.relu3_1 = nn.Sequential(*layers[7:12])
        self.relu4_1 = nn.Sequential(*layers[12:21])

        #self.apply(fix)
        
    def forward(self,inputs):
        r1 = self.relu1_1(inputs)
        r2 = self.relu2_1(r1)
        r3 = self.relu3_1(r2)
        r4 = self.relu4_1(r3)
        return r1,r2,r3,r4
