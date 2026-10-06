import torch
import torch.nn as nn


class CenterHead(nn.Module):

    def __init__(
            self,
            in_channels=256,
            num_classes=10,
            hidden_channels=256,
    ):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Conv2d(
            hidden_channels,
            num_classes,
            kernel_size=1,
        )

        self.offset_head = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
        )

        self.height_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )

        self.size_head = nn.Conv2d(
            hidden_channels,
            3,
            kernel_size=1,
        )

        self.rotation_head = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
        )

        self.velocity_head = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            heatmap:  [B, num_classes, H, W]
            offset:   [B, 2, H, W]
            height:   [B, 1, H, W]
            size:     [B, 3, H, W]
            rotation: [B, 2, H, W]
            velocity: [B, 2, H, W]
        """

        x = self.shared_conv(x)

        heatmap = self.heatmap_head(x)
        offset = self.offset_head(x)
        height = self.height_head(x)
        size = self.size_head(x)
        rotation = self.rotation_head(x)
        velocity = self.velocity_head(x)

        return {
            "heatmap": heatmap,
            "offset": offset,
            "height": height,
            "size": size,
            "rotation": rotation,
            "velocity": velocity,
        }