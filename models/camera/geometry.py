import torch
import torch.nn as nn


class GeometryTransform(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        frustum,
        intrinsics,
        extrinsics,
    ):
        """
        Args:
            frustum:
                [D, H, W, 3]

            intrinsics:
                [B, N_cam, 3, 3]

            extrinsics:
                [B, N_cam, 4, 4]

        Returns:
            points_lidar:
                [B, N_cam, D, H, W, 3]

        """ 

        B, cam_nums, _, _ =intrinsics.shape
        D, H, W, _ = frustum.shape

        frustum = frustum.view(1, 1, D, H, W, 3) # [1, 1, D, H, W, 3]
        frustum = frustum.expand(B, cam_nums, D, H, W, 3) # [B, N_cam, D, H, W, 3]

        fx = intrinsics[..., 0, 0] # [B, N_cam]
        fy = intrinsics[..., 1, 1] # [B, N_cam]
        cx = intrinsics[..., 0, 2] # [B, N_cam]
        cy = intrinsics[..., 1, 2] # [B, N_cam]

        fx = fx.reshape(B, cam_nums, 1, 1, 1) # [B, N_cam, 1, 1, 1]
        fy = fy.reshape(B, cam_nums, 1, 1, 1) # [B, N_cam, 1, 1, 1]
        cx = cx.reshape(B, cam_nums, 1, 1, 1) # [B, N_cam, 1, 1, 1]
        cy = cy.reshape(B, cam_nums, 1, 1, 1) # [B, N_cam, 1, 1, 1]

        u = frustum[..., 0] # [B, N_cam, D, H, W]
        v = frustum[..., 1] # [B, N_cam, D, H, W]
        depth = frustum[..., 2] # [B, N_cam, D, H, W]

        # 像素坐标 -> 相机坐标
        x_cam = ((u - cx) / fx) * depth # [B, N_cam, D, H, W]
        y_cam = ((v - cy) / fy) * depth # [B, N_cam, D, H, W]
        z_cam = depth # [B, N_cam, D, H, W]

        ones = torch.ones_like(z_cam) # [B, N_cam, D, H, W]
        points_cam = torch.stack([
            x_cam, y_cam, z_cam, ones,
        ], dim=-1) # [B, N_cam, D, H, W, 4]

        # 相机坐标 -> 世界坐标
        extrinsics = extrinsics.transpose(-1, -2)  # [B, N_cam, 4, 4]
        extrinsics = extrinsics.unsqueeze(2).unsqueeze(3)  # [B, N_cam, 1, 1, 1, 4, 4]
        points_lidar = torch.matmul(points_cam, extrinsics) # [B, N_cam, D, H, W, 4]

        points_lidar = points_lidar[..., :3] # [B, N_cam, D, H, W, 3] 去除齐次坐标
        return points_lidar

