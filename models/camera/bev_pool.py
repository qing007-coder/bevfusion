import torch
import torch.nn as nn


class BEVPool(nn.Module):
    def __init__(
        self,
        x_range,
        y_range,
        voxel_size,
    ):
        super().__init__()

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        self.dx = voxel_size[0]
        self.dy = voxel_size[1]

        self.bev_w = int((self.x_max - self.x_min) / self.dx)
        self.bev_h = int((self.y_max - self.y_min) / self.dy)

    def forward(self, frustum_feature, points_lidar):
        """
        Args:
            frustum_feature: frustum 空间中的特征图
                [B, N_cam, D, C, H, W]
            points_lidar: 作为 frustum_feature 对应的点云坐标
                [B, N_cam, D, H, W, 3]

        Returns:
            bev_feature: BEV 空间中的特征图
                [B, C, bev_h, bev_w]
        """

        # frustum_feature: [B, N_cam, D, C, H, W]
        B, N_cam, D, C, H, W = frustum_feature.shape

        # [B, N_cam, D, H, W, C] -> [B, M, C]   M 可以理解成点的个数
        feature = frustum_feature.permute(0, 1, 2, 4, 5, 3)
        feature = feature.reshape(B, N_cam * D * H * W, C)

        M = N_cam * D * H * W

        points = points_lidar.reshape(B, M, 3) # [B, M, 3]

        x = points[..., 0] # [B, M]
        y = points[..., 1] # [B, M]

        # 计算每个点对应的 BEV 栅格索引
        ix = torch.floor((x - self.x_min) / self.dx).long() # [B, M]
        iy = torch.floor((y - self.y_min) / self.dy).long() # [B, M]

        valid = (
            (ix >= 0) &
            (ix < self.bev_w) &
            (iy >= 0) &
            (iy < self.bev_h)
        ) # [B, M]

        bev_feature = torch.zeros(
            B, C, self.bev_h * self.bev_w,
            device=frustum_feature.device,
            dtype=frustum_feature.dtype,
        ) # [B, C, bev_h * bev_w]

        for b in range(B):
            mask = valid[b] # [M]

            if not mask.any():
                continue

            ix_b = ix[b][mask] # [M_valid]
            iy_b = iy[b][mask] # [M_valid]

            bev_flatten = iy_b * self.bev_w + ix_b # [M_valid]

            feature_b = feature[b][mask] # [M_valid, C]

            # 将特征累加到对应的 BEV 栅格
            bev_feature[b].index_add_(1, bev_flatten, feature_b.permute(1, 0)) # [C, bev_h * bev_w]

        return bev_feature.view(B, C, self.bev_h, self.bev_w)

            