#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from fpn import FPN,SSH
from mobilenet import MobileNet
from torchvision.models import _utils


class LandmarkHead(nn.Module):
    def __init__(self,inchannels = 64, num_anchors = 2):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*136,kernel_size = 1,stride = 1,padding = 0)
    
    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0],-1,136)


# only select mobilenet version to train model
class RetinaFace(nn.Module):
    def __init__(self,cfg = None,phase = 'train'):
        super(RetinaFace,self).__init__()
        self.phase = phase
        
        backbone = MobileNet()
        self.body = _utils.IntermediateLayerGetter(backbone,cfg['return_layers'])
        
        in_channels= cfg['in_channel']  # in channel is 32 (check from utils.config)
        in_channel_list = [in_channels*2,in_channels*4,in_channels*8] #[64,128,256]
        
        out_channels = cfg['out_channel'] # out channel is 64 (check from utils.config)
        
        self.fpn = FPN(in_channel_list,out_channels)
        self.feature_1 = SSH(out_channels,out_channels)
        self.feature_2 = SSH(out_channels,out_channels)
        self.feature_3 = SSH(out_channels,out_channels)
        
        self.makelandmarkhead(fpn_num = 3,inchannels = cfg['out_channel'])
        
    def makelandmarkhead(fpn_num = 3,inchannels = 64,num_anchors = 2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels = 64,num_anchors = 2))
        return landmarkhead
    
    def forward(self,inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        
        feature1 = self.feature_1(fpn[0])
        feature2 = self.feature_2(fpn[1])
        feature3 = self.feature_3(fpn[2])
        
        features = [feature1,feature2,feature3]
        ldm_regression = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)])
        
        if self.phase == 'train':
            output = ldm_regression
        return output





