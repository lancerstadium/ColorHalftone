import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from ..common.network import *


# ================= 1. 支持多通道的分类网络 =================

class AdaptiveThreshold(nn.Module):
    """可学习的阈值模块"""
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        return torch.sigmoid(10*(x - self.threshold))  # 软阈值

class PatchClassifier(nn.Module):
    def __init__(self, in_channels=3, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
        # 动态计算全连接层输入尺寸
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16x16 -> 8x8
        )
        
        # 自动计算特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, patch_size, patch_size)
            dummy_output = self.conv(dummy_input)
            fc_input_dim = dummy_output.view(-1).shape[0]
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1的概率值
        )

        self.threshold_layer = AdaptiveThreshold()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        base_output = self.fc(x)
        return self.threshold_layer(base_output)
    


