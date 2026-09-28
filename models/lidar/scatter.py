import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Scatter(nn.Module):

    def __init__(self, channel_nums, bev_shape):
        super().__init__()

        self.channel_nums = channel_nums
        self.bev_shape = bev_shape


    def forward(self, batch_size, coords, pillar_features: Tensor):
        """
        Args:
            batch_size:
                int
                当前 Batch 中样本的数量。

            coords:
                [N_total, 3]

                每个 Pillar 的坐标:
                [batch_id, pillar_y, pillar_x]

                N_total = 整个 Batch 中实际保留的
                          非空 Pillar 总数量。

            pillar_features:
                [N_total, channel_nums]

                每个 Pillar 经过 PFN 编码后的特征。

        Returns:
            bev_features:
                [batch_size, channel_nums, bev_shape[0], bev_shape[1]]

                将散乱的 Pillar 特征按 coords 中的位置
                散射回规则的 BEV 特征图上。

                每个 Pillar 特征写入对应 (batch_id, :, pillar_y, pillar_x)
                位置，未写入的位置保持为 0。
        """

        H, W = self.bev_shape

        bev_features = torch.zeros(
            batch_size, self.channel_nums, H, W,
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )  # [B, C, H, W]

        for i in range(pillar_features.shape[0]):
            batch_id = coords[i, 0].long()
            pillar_y = coords[i, 1].long()
            pillar_x = coords[i, 2].long()

            bev_features[batch_id, :, pillar_y, pillar_x] = pillar_features[i]


        return bev_features
    