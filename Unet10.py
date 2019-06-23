from config10 import device

import torch
from torch import nn
import torch.nn.functional as F


def weight_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(0.0, 1)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)


inplace = True
bias = True

base_channel = 32

def squeeze_block(base_channel):
    return nn.Sequential(
        nn.Conv2d(base_channel, 2*base_channel, 4, stride=2, padding=1),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm2d(2*base_channel),
        nn.Conv2d(2*base_channel, 2*base_channel, 3, padding=1),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm2d(2*base_channel)
    )


def pass_block(base_channel):
    return nn.Sequential(
        nn.Conv2d(base_channel, base_channel, 3, padding=1),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm2d(base_channel)
    )


def unsqueeze_block(base_channel):
    return nn.Sequential(
        nn.Conv2d(4*base_channel, 2*base_channel, 3, padding=1),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm2d(2*base_channel),
        nn.ConvTranspose2d(2*base_channel, base_channel,
                           4, stride=2, padding=1),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm2d(base_channel)
    )


def gate(base_channel):
    return nn.Sequential(
        nn.Conv2d(2*base_channel, base_channel, 3, padding=1),
        nn.Sigmoid()
    )


class BasicBlock(nn.Module):
    def __init__(self, base_channel):
        super(BasicBlock, self).__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(base_channel, base_channel, 3, padding=1),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(base_channel, base_channel, 3, padding=1)
        )

    def forward(self, x):
        residual = x
        x = self.f(x)
        return residual+x


class SqueezeBlock(nn.Module):
    def __init__(self, base_channel):
        super(SqueezeBlock, self).__init__()
        self.squeeze = squeeze_block(base_channel)
        self.trans = BasicBlock(2*base_channel)

    def forward(self, input_tensor):
        return self.trans(self.squeeze(input_tensor))


class UnsqueezeBlock(nn.Module):
    def __init__(self, base_channel):
        super(UnsqueezeBlock, self).__init__()
        self.unsqueeze1 = nn.Sequential(
            nn.ConvTranspose2d(2*base_channel, base_channel,
                               4, stride=2, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(base_channel),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2*base_channel, base_channel, 3, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(base_channel),
        )

    def forward(self, input_tensor, skip_tensor):
        t1 = self.unsqueeze1(input_tensor)
        t = torch.cat([t1, skip_tensor], 1)
        return self.fuse(t)


class GenerateBlock(nn.Module):
    def __init__(self, base_channel, out_channel):
        super(GenerateBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2*base_channel, 2*base_channel, 3, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(2*base_channel),
            nn.ConvTranspose2d(2*base_channel, base_channel,
                               4, stride=2, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(base_channel),
            nn.Conv2d(base_channel, base_channel, 3, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(base_channel),
            nn.Conv2d(base_channel, out_channel, 3, padding=1)
        )

    def forward(self, tensor):
        return self.net(tensor)


class InputNet(nn.Module):
    def __init__(self):
        super(InputNet, self).__init__()
        self.conv1_1 = nn.Sequential(  # ab hint
            nn.Conv2d(3, base_channel, 3, padding=1),
        )  # 3-32; 256-256
        self.conv1_2 = nn.Sequential(  # L hint
            nn.Conv2d(1, base_channel, 3, padding=1),
        )  # 1-32; 256-256
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*base_channel, base_channel, 3, padding=1),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(base_channel),
        )
        self.conv2 = squeeze_block(base_channel)  # 32-64; 256-128
        self.conv3 = squeeze_block(2*base_channel)  # 64-128; 128-64
        self.conv4 = squeeze_block(4*base_channel)  # 128-256; 64-32
        self.conv5 = squeeze_block(8*base_channel)  # 256-512; 32-16
        m = 16
        self.conv5_1 = nn.Conv2d(m*base_channel, m*base_channel, 3, padding=1)
        self.pass1 = BasicBlock(m*base_channel)
        self.pass2 = BasicBlock(m*base_channel)
        self.pass3 = BasicBlock(m*base_channel)
        self.pass4 = BasicBlock(m*base_channel)
        self.dconv5 = UnsqueezeBlock(8*base_channel)  # 512->256; 16-16-32
        self.dconv4 = UnsqueezeBlock(4*base_channel)  # 256-128; 32-64
        self.dconv3 = UnsqueezeBlock(2*base_channel)  # 128-64; 64-128
        self.g_l = GenerateBlock(base_channel, 1)
        self.g_ab = GenerateBlock(base_channel, 2)

        self.apply(weight_init)

    def forward(self, lineart, hint, iteration):
        c1_1 = self.conv1_1(hint)  # 3-32
        c1_2 = self.conv1_2(lineart)  # 1-32
        c1 = self.conv1(torch.cat([c1_1, c1_2], 1))  # 64-32; 256-256
        c2 = self.conv2(c1)  # 32-64; 256-128
        c3 = self.conv3(c2)  # 64-128; 128-64
        c4 = self.conv4(c3)  # 128-256; 64-32
        c5 = self.conv5(c4)  # 256-512; 32-16
        c5_1 = self.conv5_1(c5)
        p = self.pass1(c5_1)  # 512-512
        p = self.pass2(p)
        p = self.pass3(p)
        p = self.pass4(p)
        t = self.dconv5(p, c4)  # 512->256+256->256; 16-32
        t = self.dconv4(t, c3)  # 256->128+128->128; 32-64
        t = self.dconv3(t, c2)  # 128->64+64->64; 64-128
        L = self.g_l(t)  # 64-64-32-32-1
        ab = self.g_ab(t)
        img = torch.cat([L, ab], 1)
        return img
