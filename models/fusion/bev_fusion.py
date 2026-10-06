import torch
import torch.nn as nn


class BEVFusion(nn.Module):

    def __init__(self, camera_channels=256, lidar_channels=256, out_channels=256):
        super().__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(camera_channels + lidar_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_camera, bev_lidar):
        """
        Args:
            bev_camera: [B, camera_channels, H, W]
            bev_lidar:  [B, lidar_channels, H, W]

        Returns:
            bev_fused:  [B, out_channels, H, W]
        """

        # 特征融合
        bev_fused = torch.cat([bev_camera, bev_lidar], dim=1) # [B, camera_channels+lidar_channels, H, W]
        bev_fused = self.fusion_conv(bev_fused) # [B, out_channels, H, W]

        return bev_fused 