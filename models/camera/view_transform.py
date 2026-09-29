import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_net import DepthNet


class ViewTransform(nn.Module):
    """
    LSS 的 Lift 抬升模块。

    作用：
        输入 2D 图像特征，对每个像素预测深度分布，
        然后把 2D 特征“抬升”成 3D 视锥特征。

    结构：
        1. DepthNet:     对每个像素预测 depth_bins 个深度 logits
        2. softmax:      在深度维做 softmax，得到深度概率分布
        3. feature_net:  1x1 conv 生成用于抬升的特征
        4. 广播相乘:     用深度概率对特征加权，得到 5D 视锥特征

    输入：
        img_features: [B*N_cam, in_channels, H, W]

    输出：
        frustum_feature: [B*N_cam, depth_bins, feature_channels, H, W]

    说明：
        输出的 5D 张量中：
            dim 0: B*N_cam        batch × 相机数
            dim 1: depth_bins     深度 bin 数
            dim 2: feature_channels 特征通道
            dim 3: H              特征图高
            dim 4: W              特征图宽
        后续 frustum / bev_pool 会利用这个结构把特征撒到 BEV 网格上。
    """

    def __init__(self, depth_bins=80, in_channels=256,
                 mid_channels=256, feature_channels=256):
        """
        Args:
            depth_bins:       深度离散化的 bin 数，例如 80
            in_channels:      输入图像特征的通道数，通常等于 FPN 的 out_channels
            mid_channels:     DepthNet 中间层通道数
            feature_channels: 用于抬升的特征通道数
        """
        super().__init__()

        # 深度预测网络：输出每个像素在 depth_bins 上的 logits
        self.depth_net = DepthNet(
            depth_bins=depth_bins,
            in_channels=in_channels,
            mid_channels=mid_channels,
        )

        # 特征变换：1x1 conv，把输入特征映射到 feature_channels
        self.feature_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=1,
        )

    def forward(self, img_features):
        """
        Args:
            img_features: [B*N_cam, in_channels, H, W]
                          B*N_cam: batch × 相机数
                          H, W:    特征图高宽（已下采样过的）

        Returns:
            frustum_feature: [B*N_cam, depth_bins, feature_channels, H, W]
        """

        depth_logits = self.depth_net(img_features) # [B*N_cam, depth_bins, H, W]

        depth_prob = F.softmax(depth_logits, dim=1) # [B*N_cam, depth_bins, H, W]

        feature = self.feature_net(img_features) # [B*N_cam, feature_channels, H, W]

        depth_prob = depth_prob.unsqueeze(2) # [B*N_cam, depth_bins, 1, H, W]

        feature = feature.unsqueeze(1) # [B*N_cam, 1, feature_channels, H, W]

        # 广播相乘：深度维 × 通道维 的外积
        frustum_feature = depth_prob * feature # [B*N_cam, depth_bins, feature_channels, H, W]

        return frustum_feature