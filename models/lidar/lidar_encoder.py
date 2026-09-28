import torch
import torch.nn as nn
from torch import Tensor

from .pillar_feature import PillarFeatureBuilder
from .pfn import PFNLayer
from .pillarizer import Pillarizer
from .scatter import Scatter


class LidarEncoder(nn.Module):

    """
        PointPillars 的 Lidar 编码器。

        把原始点云编码成 BEV 特征图。

        流程:
            points
              -> Pillarizer           划分 pillar  得到 pillars 和 coords
              -> PillarFeatureBuilder 构造 9 维点特征
              -> PFNLayer             逐点编码 + pillar 内 max pooling
              -> Scatter              散射回 BEV 特征图

        输入:
            points: [B, N, 3] 或 [B, N, 4]

        输出:
            bev_features: [B, out_channels, bev_h, bev_w]
    """

    def __init__(
            self,
            point_cloud_range,
            voxel_size,
            max_points_per_pillar=32,
            max_pillars=12000,
            in_channels=9,
            out_channels=64,
    ):
        super().__init__()

        self.pillarizer = Pillarizer(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )

        self.pillar_feature_builder = PillarFeatureBuilder(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )

        self.pfn = PFNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        x_min, y_min, _, x_max, y_max, _ = point_cloud_range
        vx, vy = voxel_size[0], voxel_size[1]

        bev_h = int((y_max - y_min) / vy)
        bev_w = int((x_max - x_min) / vx)
        self.bev_shape = (bev_h, bev_w)

        self.scatter = Scatter(
            channel_nums=out_channels,
            bev_shape=self.bev_shape,
        )

    def forward(self, points):
        """
        Args:
            points:
                [B, N, 4]

                B = batch size
                N = 每个样本的点数
                最后一维是点的特征，至少包含 x, y, z

        Returns:
            bev_features:
                [B, out_channels, bev_h, bev_w]

                编码后的 BEV 特征图。
        """

        B = points.shape[0]

        pillars, coords = self.pillarizer(points) # [N_total, max_points_per_pillar, 4], [N_total, 3]
        pillar_features = self.pillar_feature_builder(pillars, coords)

        pillar_features = self.pfn(pillar_features)

        bev_features = self.scatter(B, coords, pillar_features)
        return bev_features
    