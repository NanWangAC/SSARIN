import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from logging.handlers import RotatingFileHandler
from turtle import forward
import torch.nn as nn
import torch
from Rotate import *
import numpy as np
import torch
from torch import nn
from mmcv.cnn import ConvModule

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out_1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out_2 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out_1 + avg_out_2
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HyperNet(nn.Module):
    def __init__(self, HSI_Num):
        super(HyperNet, self).__init__()
        self.bs = ChannelAttention(in_planes=HSI_Num)
        self.SpatialAttention1 = SpatialAttention()
        self.SpatialAttention2 = SpatialAttention()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=HSI_Num, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels =128, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels =256, out_channels=512, kernel_size=(3, 3),padding=(1,1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels =512, out_channels=256, kernel_size=(5, 5),padding=(1,1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels =256, out_channels=128, kernel_size=(3, 3),padding=(1,1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),           
            nn.Conv2d(in_channels =128, out_channels=64, kernel_size=(1, 1),),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))                    

        self.feature_enhance = nn.Sequential(
             nn.Conv2d(in_channels =64, out_channels=256, kernel_size=(1, 1)),                     
             nn.ReLU(inplace=True),
             nn.Conv2d(in_channels =256, out_channels=64, kernel_size=(1, 1)),
             nn.ReLU(inplace=True),
             nn.AdaptiveAvgPool2d(1))

        self.Classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.LogSoftmax(dim=1)
        )
        

    def forward(self, x):
        x = x*self.bs(x)
        
        x0 = x
        x1 = rotate(x, stride=1)
        x2 = rotate(x, stride=2)
        x3 = rotate(x, stride=3)
        x4 = rotate(x, stride=4)
        x5 = rotate(x, stride=5)
        x6 = rotate(x, stride=6)
        x7 = rotate(x, stride=7)

        x0 = self.conv1(x0)
        x0 = x0*self.SpatialAttention1(x0)
        x0 = self.conv2(x0)
        x0 = x0*self.SpatialAttention2(x0)
        x0 = self.conv4(x0)  
        
        x1 = self.conv1(x1)
        x1 = x1*self.SpatialAttention1(x1)
        x1 = self.conv2(x1)
        x1 = x1*self.SpatialAttention2(x1)
        x1 = self.conv4(x1)

        x2 = self.conv1(x2)
        x2 = x2*self.SpatialAttention1(x2)
        x2 = self.conv2(x2)
        x2 = x2*self.SpatialAttention2(x2)
        x2 = self.conv4(x2)

        x3 = self.conv1(x3)
        x3 = x3*self.SpatialAttention1(x3)
        x3 = self.conv2(x3)
        x3 = x3*self.SpatialAttention2(x3)
        x3 = self.conv4(x3)
        
        x4 = self.conv1(x4)
        x4 = x4*self.SpatialAttention1(x4)
        x4 = self.conv2(x4)
        x4 = x4*self.SpatialAttention2(x4)
        x4 = self.conv4(x4)
        
        x5 = self.conv1(x5)
        x5 = x5*self.SpatialAttention1(x5)
        x5 = self.conv2(x5)
        x5 = x5*self.SpatialAttention2(x5)
        x5 = self.conv4(x5)
        
        x6 = self.conv1(x6)
        x6 = x6*self.SpatialAttention1(x6)
        x6 = self.conv2(x6)
        x6 = x6*self.SpatialAttention2(x6)
        x6 = self.conv4(x6)
        
        x7 = self.conv1(x7)
        x7 = x7*self.SpatialAttention1(x7)
        x7 = self.conv2(x7)
        x7 = x7*self.SpatialAttention2(x7)
        x7 = self.conv4(x7)

        x = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7)/8
        x = self.feature_enhance(x)
        x = x.contiguous().view(x.size(0),-1)
        x = self.Classifier(x)
        return x
