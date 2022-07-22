#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

# kernel_size = 3 conv layer (cbl module)
def conv_bn(inp,outp,stride = 1,leaky = 0):
    return nn.Sequential(nn.Conv2d(inp,outp,3,stride,1,bias = False),
                        nn.BatchNorm2d(outp),
                        nn.LeakyReLU(negative_slope = leaky,inplace = True))


# 1*1 kernel conv layer
def conv_bn1x1(inp,outp,stride,leaky = 0):
    return nn.Sequential(nn.Conv2d(inp,outp,1,stride,padding = 0,bias = False),
                        nn.BatchNorm2d(outp),
                        nn.LeakyReLU(negative_slope = leaky,inplace = True))

# conv layer without relu
def conv_bn_no_relu(inp,outp,stride):
    return nn.Sequential(nn.Conv2d(inp,outp,3,stride,1,bias = False),
                        nn.BatchNorm2d(outp))


class FPN(nn.Module):
    def __init__(self,in_channel_list,out_channel):
        super(FPN,self).__init__()
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.output1 = conv1x1(in_channel_list[0],out_channel,stride = 1,leaky = leaky)
        self.output2 = conv1x1(in_channel_list[1],out_channel,stride = 1,leaky = leaky)
        self.output3 = conv1x1(in_channel_list[2],out_channel,stride = 1,leaky = leaky)
        
        self.merge1 = conv_bn(out_channel,out_channel,leaky = leaky)
        self.merge2 = conv_bn(out_channel,out_channel,leaky = leaky)
    
    def forward(self,inputs):
        # ------------------------
        # inputs are features with sizes : 
        # C3:80,80,64
        # C4:40,40,128
        # C5:20,20,256
        #-------------------------------
        inputs = list(inputs.value())
        
        #------------------------------------------------------------
        # we can get output feature : 80,80,64;  40,40,64;  20,20,64
        #------------------------------------------------------------
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])
        
        # up sample and feature merge
        up3 = F.interpolate(output3,size = [output2.size(2),output2.size(3)],mode = 'nearest')
        output2 = up3 + output2
        output2 = self.merge2(output2)
        
        up2 = F.interpolate(output2,size = [output1.size(2),output1.size(3)],mode = 'nearest')
        output1 = up2 + output1
        output1 = self.merge1(output1)
        
        out = [output1,output2,output3]
        return out
        
        

class SSH(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(self,SSH).__init__()
        assert out_channel%4 ==0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        # 3*3 conv layer
        self.conv3x3 = conv_bn_no_relu(in_channel,out_channel//2,stride = 1)
        #5*5 conv layer --> use 2 3*3 kernel to replace 5*5 kernel
        self.conv5x5_1 = conv_bn(in_channel,out_channel//4,stride = 1,leaky = leaky)
        self.conv5x5_2 = conv_bn_no_relu(out_channel//4,out_channel//4,stride = 1)
        # 7*7 conv layer --> use 3 3*3 kernel to replace 7*7 kernel
        self.conv7x7_2 = conv_bn(in_channel,out_channel//4,stride = 1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4,out_channel//4,stride = 1)
        
    def forward(self,inputs):
        feature_1 = self.conv3x3(inputs)
        
        feature_2_1 = self.conv5x5_1(inputs)
        feature_2 = self.conv5x5_2(feature_2_1)
        
        feature_3 = self.conv7x7_2(feature_2_1)
        feature_3 = self.conv7x7_3(feature_3)
        
        out = torch.cat([feature_1,feature_2,feature_3], dim = 1)
        out = F.relu(out)
        return out






