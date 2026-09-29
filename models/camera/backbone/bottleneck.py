import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, mid_channels, stride, downsample):
        super().__init__()

        self.out_channels = mid_channels * self.expansion

        # 降维
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, stride=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 特征提取 下采样 
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, stride=stride, padding=1, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 升维
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=self.out_channels, stride=1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)

        residual = self.conv3(residual)
        residual = self.bn3(residual)

        # 防止残差和x维度对不上
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = identity + residual
        out = self.relu(out)
        return out

