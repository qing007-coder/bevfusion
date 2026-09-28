import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PFNLayer(nn.Module):

    def __init__(
            self,
            in_channels=9,
            out_channels=64,
    ):
        super().__init__()

        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor):
        """
        Args:
            x: [N_total, max_points_per_pillar, 9]

        Returns:
            ouput: [N_total, 64]    
        """
        N, P, _ = x.shape

        x = self.linear(x) # [N_total, max_points_per_pillar, 64]

        # 用明确的通道数 reshape:
        # N == 0 (整个 Batch 没有有效 Pillar) 时 -1 无法推断。
        x = x.reshape(N * P, self.out_channels)
        x = self.bn(x)
        x = x.reshape(N, P, self.out_channels)

        x = F.relu(x) # [N_total, max_points_per_pillar, 64]

        output, _ = torch.max(x, dim=1) # [N_total, 64]

        return output