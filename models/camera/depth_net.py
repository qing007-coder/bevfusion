import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):

    def __init__(self, depth_bins=80, in_channels=256, mid_channels=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.depth_head = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=depth_bins,
            kernel_size=1,
        )

    def forward(self, img_features):
        """
        Args:
            img_features [B*N_cam, 256, H, W]
        """

        img_features = self.conv(img_features)

        depth_logits = self.depth_head(img_features)

        return depth_logits
