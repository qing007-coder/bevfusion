import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.resnet import ResNet
from .backbone.fpn import FPN


class CameraEncoder(nn.Module):
    """
    相机图像编码器。

    负责把多相机原始图像编码成多尺度 2D 特征，
    供后续 depth / frustum / view transform / bev pool 使用。

    数据流：
        images  ->  ResNet  ->  (c2, c3, c4, c5)  ->  FPN  ->  (p2, p3, p4, p5)

    注意：
        - 相机维 cam_nums 会被合并到 batch 维一起送进 backbone，
          所以输出的每个尺度都是 [B*cam_nums, C, H', W']。
        - 相机维的还原（reshape 回 [B, cam_nums, ...]）由下游模块处理。
    """

    def __init__(self, layers, in_channels, out_channels):
        """
        Args:
            layers:       ResNet 每个 stage 的 Bottleneck 数量，例如 ResNet-50 是 [3, 4, 6, 3]。
            in_channels:  ResNet 四个 stage 的输出通道，传给 FPN 的 lateral 分支使用，
                          通常为 (256, 512, 1024, 2048)。
            out_channels: FPN 每一层统一输出的通道数，通常为 256。
        """
        super().__init__()

        # 图像骨干网络：提取 C2 ~ C5 多尺度特征
        self.backbone = ResNet(layers)

        # 特征金字塔：把 C2 ~ C5 融合成通道统一的 P2 ~ P5
        self.neck = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, images):
        """
        Args:
            images: [B, cam_nums, C, H, W]
                    B         batch size
                    cam_nums  相机数量（环视相机数）
                    C         图像通道数，通常为 3
                    H, W      输入图像高宽

        Returns:
            p2: [B*cam_nums, out_channels, H/4,  W/4 ]
            p3: [B*cam_nums, out_channels, H/8,  W/8 ]
            p4: [B*cam_nums, out_channels, H/16, W/16]
            p5: [B*cam_nums, out_channels, H/32, W/32]

            说明：
                - 相机维尚未还原  reshape 回 [B, cam_nums, ...] 由下游负责。
                - 四个尺度的通道数都已统一为 out_channels。
        """

        B, cam_nums, C, H, W = images.shape
        images = images.reshape(B * cam_nums, C, H, W)

        c2, c3, c4, c5 = self.backbone(images)

        multi_scale_features = self.neck(c2, c3, c4, c5)

        return multi_scale_features