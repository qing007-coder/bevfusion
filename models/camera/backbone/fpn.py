import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)

    自顶向下的特征金字塔，把 ResNet 输出的多尺度特征 C2 ~ C5
    融合成统一通道数的 P2 ~ P5 供后续检测 / 分割 / BEV 等任务使用。

    结构：
        - lateral connection 横向连接 : 1x1 conv 把 C_i 的通道对齐到 out_channels
        - top-down pathway 自顶向下 :  高层特征上采样后与低层相加
        - smooth 平滑:               3x3 conv 消除上采样带来的锯齿/混叠

    对应关系：
        C2 -> P2   (stride 4,  分辨率最高)
        C3 -> P3   (stride 8)
        C4 -> P4   (stride 16)
        C5 -> P5   (stride 32, 分辨率最低、语义最强)
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        """
        Args:
            in_channels:  ResNet 四个 stage 的输出通道，对应 C2 ~ C5
            out_channels: FPN 每一层统一输出的通道数，通常为 256
        """
        super().__init__()

        # Lateral connections: 1x1 conv 把 C_i 通道对齐到 out_channels
        self.lateral_c2 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels,
            kernel_size=1,
        )
        self.lateral_c3 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels,
            kernel_size=1,
        )
        self.lateral_c4 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels,
            kernel_size=1,
        )
        self.lateral_c5 = nn.Conv2d(
            in_channels=in_channels[3], out_channels=out_channels,
            kernel_size=1,
        )

        # Smooth: 3x3 conv，融合后做平滑，保持通道和尺寸不变
        self.smooth_c2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1,
        )
        self.smooth_c3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1,
        )
        self.smooth_c4 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1,
        )
        self.smooth_c5 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, c2, c3, c4, c5):
        """
        Args:
            c2: [B, in_channels[0], H/4,  W/4]
            c3: [B, in_channels[1], H/8,  W/8]
            c4: [B, in_channels[2], H/16, W/16]
            c5: [B, in_channels[3], H/32, W/32]

        Returns:
            p2: [B, out_channels, H/4,  W/4]
            p3: [B, out_channels, H/8,  W/8]
            p4: [B, out_channels, H/16, W/16]
            p5: [B, out_channels, H/32, W/32]
        """

        p5 = self.lateral_c5(c5)          # [B, out_channels, H/32, W/32]
        p5 = self.smooth_c5(p5)           # [B, out_channels, H/32, W/32]

        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="nearest"
        )
        p4 = self.smooth_c4(p4)           # [B, out_channels, H/16, W/16]

        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="nearest"
        )
        p3 = self.smooth_c3(p3)           # [B, out_channels, H/8, W/8]

        p2 = self.lateral_c2(c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="nearest"
        )
        p2 = self.smooth_c2(p2)           # [B, out_channels, H/4, W/4]

        return p2, p3, p4, p5