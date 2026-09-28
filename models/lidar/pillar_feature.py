import torch
import torch.nn as nn
from torch import Tensor


class PillarFeatureBuilder(nn.Module):
    """
    构造 PointPillars 的 9 维点特征。

    原始:
        x, y, z, intensity

    增加:
        x - x_mean 点对所有点x均值的偏移
        y - y_mean 点对所有点y均值的偏移
        z - z_mean 点对所有点z均值的偏移

        x - x_center 点对柱体重心x的偏移
        y - y_center 点对柱体重心y的偏移

    最终:
        9维
    """
    def __init__(
            self,
            point_cloud_range,
            voxel_size,
    ):
        super().__init__()

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]

        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]

    def forward(self, pillars: Tensor, coords):
        """
        Args:
            pillars:
                [N_total, max_points_per_pillar, 4]

                N_total = 整个 Batch 中实际保留的
                          非空 Pillar 总数量。

            coords:
                [N_total, 3]

                每个 Pillar 的坐标:
                [batch_id, pillar_y, pillar_x]

        Returns:
            features:
                [N_total, max_points_per_pillar, 9]            
        """

        xyz = pillars[:, :, :3] # [N_total, max_points_per_pillar, 3]

        xyz_mean = xyz.mean(dim=1, keepdim=True) # [N_total, 1, 3]
        xyz_offset = xyz - xyz_mean # [N_total, max_points_per_pillar, 3]

        pillars_y = coords[:, 1:2].float()  # [N_total, 1]  这样写可以保留切片
        pillars_x = coords[:, 2:3].float()  # [N_total, 1]  

        pillars_x_center = (
            pillars_x * self.vx + self.vx * 0.5 + self.x_min
        ) # [N_total, 1]

        pillars_y_center = (
            pillars_y * self.vy + self.vy * 0.5 + self.y_min
        ) # [N_total, 1]


        pillars_x_center = pillars_x_center.unsqueeze(1) # [N_total, 1, 1]
        pillars_y_center = pillars_y_center.unsqueeze(1) # [N_total, 1, 1]

        center_x_offset = (
            pillars[:, :, 0:1] - pillars_x_center
        ) # [N_total, max_points_per_pillar, 1]

        center_y_offset = (
            pillars[:, :, 1:2] - pillars_y_center
        ) # [N_total, max_points_per_pillar, 1]

        feature = torch.cat([pillars, xyz_offset, center_x_offset, center_y_offset], dim=-1) # [N_total, max_points_per_pillar, 9]

        return feature