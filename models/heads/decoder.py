import torch
import torch.nn as nn


class CenterPointDecoder(nn.Module):

    def __init__(
        self,
        top_k=100,
        voxel_size=(0.5, 0.5),
        point_cloud_range=(-50.0, -50.0),
        score_threshold=0.1,
    ):
        super().__init__()

        self.top_k = top_k

        self.voxel_size_x = voxel_size[0]
        self.voxel_size_y = voxel_size[1]

        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]

        self.score_threshold = score_threshold

    def topk(self, heatmap):
        """
        Args:
            heatmap: [B, C, H, W]

        Returns:
            topk_scores: [B, K]
            topk_classes: [B, K]
            topk_ys: [B, K]
            topk_xs: [B, K]
        """

        B, C, H, W = heatmap.shape

        heatmap = heatmap.view(B, C, -1) # [B, C, H*W]

        K = min(self.top_k, H * W)
        topk_scores, topk_indices = torch.topk(heatmap, K, dim=2) # [B, C, K]

        topk_scores = topk_scores.reshape(B, -1) # [B, C*K]
        topk_indices = topk_indices.reshape(B, -1) # [B, C*K]

        final_scores, final_indices = torch.topk(topk_scores, K, dim=1) # [B, K]

        classes = final_indices // K # [B, K]

        spatial_indices = topk_indices.gather(1, final_indices) # [B, K]
        spatial_ys = spatial_indices // W # [B, K]
        spatial_xs = spatial_indices % W # [B, K]


        return final_scores, classes, spatial_ys, spatial_xs

    def gather_feature(
        self,
        feature,
        xs,
        ys,
    ):

        """
        将topK的特征点从feature中取出
        Args:
            feature:
                [B, C, H, W]

            xs:
                [B, K]

            ys:
                [B, K]

        Returns:
            gathered:
                [B, K, C]
        """

        B, C, H, W = feature.shape

        feature = feature.view(
            B,
            C,
            -1
        ) # [B,C,H*W]

        # 转成 flatten index
        indices = ys * W + xs # [B,K]

        indices = indices.unsqueeze(1)
        indices = indices.expand(
            B,
            C,
            -1
        ) # [B,C,K]

        gathered = torch.gather(
            feature,
            2,
            indices
        ) # [B,C,K]

        gathered = gathered.permute(
            0,
            2,
            1
        ) # [B,K,C]

        return gathered



    def decode(
        self,
        outputs,
    ):

        heatmap = outputs["heatmap"]
        offset = outputs["offset"]
        height = outputs["height"]
        size = outputs["size"]
        rotation = outputs["rotation"]
        velocity = outputs["velocity"]

        heatmap = torch.sigmoid(heatmap)

        scores, classes, ys, xs = self.topk(
            heatmap
        ) # 取出 topK 的分数、类别、坐标


        # 获取 topK 的特征点
        offset = self.gather_feature(
            offset,
            xs,
            ys
        )

        height = self.gather_feature(
            height,
            xs,
            ys
        )

        size = self.gather_feature(
            size,
            xs,
            ys
        )

        rotation = self.gather_feature(
            rotation,
            xs,
            ys
        )

        velocity = self.gather_feature(
            velocity,
            xs,
            ys
        )

        xs = xs.float()
        ys = ys.float()

        # 加上 offset
        xs = xs + offset[..., 0]
        ys = ys + offset[..., 1]


        # BEV 网格 → 世界 / LiDAR 坐标
        x = xs * self.voxel_size_x + self.x_min
        y = ys * self.voxel_size_y + self.y_min

        # z轴高度
        z = height[..., 0]

        # box size
        w = size[..., 0]
        l = size[..., 1]
        h = size[..., 2]

        # rotation 朝向
        sin_yaw = rotation[..., 0]
        cos_yaw = rotation[..., 1]

        yaw = torch.atan2(
            sin_yaw,
            cos_yaw
        )

        # velocity 速度
        vx = velocity[..., 0]
        vy = velocity[..., 1]

        # score filtering
        mask = scores > self.score_threshold

        boxes = torch.stack(
            [
                x,
                y,
                z,
                w,
                l,
                h,
                yaw,
                vx,
                vy,
            ],
            dim=-1
        )

        # [B,K,9]

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "mask": mask,
        }

    def forward(self, outputs):

        return self.decode(outputs)