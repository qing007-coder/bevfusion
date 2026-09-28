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

        x = x.view(N * P, -1)
        x = self.bn(x)
        x = x.view(N, P, -1)

        x = F.relu(x) # [N_total, max_points_per_pillar, 64]

        output, _ = torch.max(x, dim=1) # [N_total, 64]

        return output