#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

def conv_bn(in_channel,out_channel,stride =1):
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,3,stride,1,bias = False),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())


def conv_dw(in_channel,out_channel,stride = 1, leaky = 0.1):
    return nn.Sequential(nn.Conv2d(in_channel,in_channel,3,stride,1,groups = in_channel,bias = False),
                         nn.BatchNorm2d(in_channel),
                         nn.LeakyReLU(negative_slope = leaky,inplace = True),
                         nn.Conv2d(in_channel,out_channel,1,1,0,bias = False),
                         nn.BatchNorm2d(out_channel),
                         nn.LeakyReLU(negative_slope = leaky,inplace = True)
              )

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.stage1 = nn.Sequential(conv_bn(3,8,2),
                                    conv_dw(8,16,1),
                                    conv_dw(16,32,1),
                                    conv_dw(32,32,1),
                                    conv_dw(32,64,1),
                                    conv_dw(64,64,1))
        self.stage2 = nn.Sequential(conv_dw(64,128,2),
                                    conv_dw(128,128,1),
                                    conv_dw(128,128,1),
                                    conv_dw(128,128,1),
                                    conv_dw(128,128,1),
                                    conv_dw(128,128,1))
        self.stage3 = nn.Sequential(conv_dw(128,256,2),
                                    conv_dw(256,256,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)),
        self.fc = nn.Linear(256,1000)
    
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(-1,256)
        x = self.fc(x)
        return x

mobilenet = MobileNet()
print(mobilenet)







