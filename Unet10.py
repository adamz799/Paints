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

        # LSTM part
        # self.conv_i = gate(16*base_channel)#input gate
        # self.conv_f = gate(16*base_channel)#forget gate
        # self.conv_o = gate(16*base_channel)#output gate
        # self.conv_c_ = nn.Sequential(
        #     nn.Conv2d(32*base_channel,16*base_channel,3,padding=1),
        #     nn.Tanh()
        # )

        # init
        # for name, modules in self.named_children():
        #     if name!='fnet':
        #         modules.apply(weight_init)
        # base_net_dict = torch.load('with_cue_v6_2.pkl').state_dict()
        # self_dict = self.state_dict()
        # base_net_dict = {name:weight for name,weight in base_net_dict.items() if name in self_dict}
        # self_dict.update(base_net_dict)
        # self.load_state_dict(self_dict)
        self.apply(weight_init)

    # def forward(self, lineart, hint, iteration):
    #     l = []
    #     #img = torch.cat([lineart, lineart, hint],1)
    #     c1_1 = self.conv1_1(hint)  # 3-32
    #     l.append(c1_1)
    #     c1_2 = self.conv1_2(lineart)  # 1-32
    #     l.append(c1_2)
    #     c1 = self.conv1(torch.cat([c1_1, c1_2], 1))  # 64-32; 256-256
    #     l.append(c1)
    #     c2 = self.conv2(c1)  # 32-64; 256-128
    #     l.append(c2)
    #     c3 = self.conv3(c2)  # 64-128; 128-64
    #     l.append(c3)
    #     c4 = self.conv4(c3)  # 128-256; 64-32
    #     l.append(c4)
    #     c5 = self.conv5(c4)  # 256-512; 32-16
    #     l.append(c5)
    #     c5_1 = self.conv5_1(c5)
    #     l.append(c5_1)
    #     p = self.pass1(c5_1)  # 512-512
    #     p = self.pass2(p)
    #     p = self.pass3(p)
    #     p = self.pass4(p)
    #     l.append(p)
    #     t = self.dconv5(p, c4)  # 512->256+256->256; 16-32
    #     t = self.dconv4(t, c3)  # 256->128+128->128; 32-64
    #     t = self.dconv3(t, c2)  # 128->64+64->64; 64-128
    #     l.append(t)
    #     L = self.g_l(t)  # 64-64-32-32-1
    #     ab = self.g_ab(t)
    #     # t1 = self.dconv2_1(t, c1_2)
    #     # t2 = self.dconv2_2(t, c1_2)# 128->64+64->64
    #     # #t = self.dconv1(t, c1)
    #     # L = self.dconv1_2(t2)
    #     # ab = self.dconv1_1(t1)
    #     img = torch.cat([L, ab], 1)
    #     return img, l

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
        # t1 = self.dconv2_1(t, c1_2)
        # t2 = self.dconv2_2(t, c1_2)# 128->64+64->64
        # #t = self.dconv1(t, c1)
        # L = self.dconv1_2(t2)
        # ab = self.dconv1_1(t1)
        img = torch.cat([L, ab], 1)
        return img

        # temp_img = rgb_batch
        # output_img = []
        # c = None
        # h = None
        # for i in range(iteration):
        #     image = torch.cat([gray_image, hint],1)#2+3
        #     if i>0:
        #         temp_img = tensor_Lab2RGB(self.d1)
        #         temp_img = rgb2torch_input_rgb(temp_img)

        #     self.r1 = self.conv1(image)#5-32; 256-256
        #     self.r2 = self.conv2(self.r1)#32-64; 256-128
        #     self.r3 = self.conv3(self.r2)#64-128; 128-64
        #     self.r4 = self.conv4(self.r3)#128-256; 64-32
        #     self.r5 = self.conv5(self.r4)#256-512; 32-16
        #     self.r6 = self.conv6(self.r5)#512-1024; 16-8
        #     self.f = self.fnet(temp_img)#[1,2048,8,8]
        #     self.d6 = self.dconv6_1(torch.cat([self.r6, self.f],1))#3072-512

        #     if i==0:
        #         c = torch.zeros(self.d6.shape).to(device)
        #         h = torch.zeros(self.d6.shape).to(device)

        #     x = torch.cat([self.d6, h],1)
        #     f = self.conv_f(x)
        #     i = self.conv_i(x)
        #     c_ = self.conv_c_(x)
        #     o = self.conv_o(x)
        #     c = f*c + i*c_
        #     h = o*F.tanh(c)

        #     self.d5 = self.dconv5(torch.cat([h, self.r5],1))#1024-256
        #     self.d4 = self.dconv4(torch.cat([self.d5, self.r4],1))#512-128
        #     self.d3 = self.dconv3(torch.cat([self.d4, self.r3],1))#256-64
        #     self.d2 = self.dconv2(torch.cat([self.d3, self.r2],1))#128-32
        #     self.d1 = self.dconv1(self.d2)#32-3

        #     output_img.append(self.d1)

        # return output_img
