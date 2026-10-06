import torch
import torch.nn as nn

from .camera.bev_pool import BEVPool
from .camera.camera_encoder import CameraEncoder
from .camera.frustum import FrustumGenerator
from .camera.geometry import GeometryTransform
from .camera.view_transform import ViewTransform
from .fusion.bev_fusion import BEVFusion
from .heads.decoder import CenterPointDecoder
from .heads.detection_head import CenterHead
from .lidar.lidar_encoder import LidarEncoder


class BEVFusionModel(nn.Module):
    """
    BEVFusion 总模型：把 LiDAR 分支、Camera 分支、融合模块和检测头串成一条完整通路。

    整体数据流:

        points [B, N, 4]
            -> LidarEncoder                       -> bev_lidar  [B, C_lidar, H_bev, W_bev]

        images [B, N_cam, 3, H, W]
            -> CameraEncoder  (ResNet + FPN, 取 P2)  -> p2  [B*N_cam, C_cam, h, w]
            -> ViewTransform  (Lift)                 -> frustum_feature [B*N_cam, D, C_cam, h, w]
            -> reshape                               -> [B, N_cam, D, C_cam, h, w]
            -> FrustumGenerator                      -> frustum [D, h, w, 3]   (u, v, depth)
            -> GeometryTransform (Splat 前的反投影)   -> points_lidar [B, N_cam, D, h, w, 3]
            -> BEVPool                               -> bev_camera [B, C_cam, H_bev, W_bev]

        bev_camera + bev_lidar
            -> BEVFusion                             -> bev_fused [B, C_fusion, H_bev, W_bev]

        bev_fused
            -> CenterHead                            -> 各检测分支的 dense 预测

    说明:
        - 两个分支输出的 BEV 特征在同一个栅格坐标系下（都是 BEV 俯视图，
          H 对应 y 方向，W 对应 x 方向），所以可以直接在通道维拼接。
        - forward 只做前向推理；训练时的 target / loss 由
          CenterPointTargetGenerator 和 CenterPointLoss 在外部组合，
          见 tests/test_full_pipeline.py。
    """

    def __init__(
            self,
            point_cloud_range=(0.0, -20.0, -3.0, 40.0, 20.0, 1.0),
            voxel_size=(0.5, 0.5, 4.0),
            max_points_per_pillar=32,
            max_pillars=12000,
            lidar_channels=64,
            resnet_layers=(3, 4, 6, 3),
            resnet_channels=(256, 512, 1024, 2048),
            camera_channels=64,
            depth_bins=80,
            depth_min=1.0,
            depth_max=80.0,
            fusion_channels=128,
            num_classes=3,
            hidden_channels=128,
            top_k=100,
            score_threshold=0.1,
    ):
        """
        Args:
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]，单位 m。
                               LiDAR 分支和 Camera 分支共用，保证两个 BEV 网格对齐。
            voxel_size:        [vx, vy, vz]，BEV 栅格分辨率，单位 m。
            max_points_per_pillar / max_pillars: 见 Pillarizer。
            lidar_channels:    LiDAR 分支输出的 BEV 通道数。
            resnet_layers:     ResNet 每个 stage 的 Bottleneck 数量。
            resnet_channels:   ResNet 四个 stage 的输出通道，给 FPN 的 lateral 用。
            camera_channels:   Camera 分支统一通道数（FPN 输出 = ViewTransform 输入 = BEV 通道数）。
            depth_bins:        深度离散化 bin 数（Lift 的 D 维）。
            depth_min/max:     深度范围，单位 m。
            fusion_channels:   融合后 BEV 的通道数。
            num_classes:       检测类别数。
            hidden_channels:   检测头共享卷积的通道数。
            top_k:             解码时每帧保留的候选框数量。
            score_threshold:   解码时的分数阈值。
        """
        super().__init__()

        self.point_cloud_range = tuple(point_cloud_range)
        self.voxel_size = tuple(voxel_size)
        self.num_classes = num_classes
        self.depth_bins = depth_bins
        self.camera_channels = camera_channels
        self.lidar_channels = lidar_channels
        self.fusion_channels = fusion_channels

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        vx, vy = self.voxel_size[0], self.voxel_size[1]

        # BEV 特征图尺寸：H 对应 y 方向，W 对应 x 方向
        self.bev_h = int((y_max - y_min) / vy)
        self.bev_w = int((x_max - x_min) / vx)

        # ------------------------------
        # LiDAR 分支
        # ------------------------------
        self.lidar_encoder = LidarEncoder(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            in_channels=9,
            out_channels=lidar_channels,
        )

        # ------------------------------
        # Camera 分支
        # ------------------------------
        self.camera_encoder = CameraEncoder(
            layers=list(resnet_layers),
            in_channels=tuple(resnet_channels),
            out_channels=camera_channels,
        )

        self.view_transform = ViewTransform(
            depth_bins=depth_bins,
            in_channels=camera_channels,
            mid_channels=camera_channels,
            feature_channels=camera_channels,
        )

        # FPN 的 P2 相对输入图像下采样 4 倍，所以 stride=4
        self.frustum_generator = FrustumGenerator(
            depth_bins=depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            stride=4,
        )

        self.geometry_transform = GeometryTransform()

        self.bev_pool = BEVPool(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            voxel_size=(vx, vy),
        )

        # ------------------------------
        # 融合
        # ------------------------------
        self.fusion = BEVFusion(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            out_channels=fusion_channels,
        )

        # ------------------------------
        # 检测头
        # ------------------------------
        self.head = CenterHead(
            in_channels=fusion_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )

        self.decoder = CenterPointDecoder(
            top_k=top_k,
            voxel_size=(vx, vy),
            point_cloud_range=(x_min, y_min),
            score_threshold=score_threshold,
        )

    # ==========================================================
    # Camera 分支
    # ==========================================================

    def encode_camera(
            self,
            images,
            intrinsics,
            extrinsics,
    ):
        """
        Args:
            images:
                [B, N_cam, 3, H, W]

            intrinsics:
                [B, N_cam, 3, 3]
                相机内参 K。

            extrinsics:
                [B, N_cam, 4, 4]
                相机 -> LiDAR 的外参 T，满足 p_lidar = T @ p_cam。

        Returns:
            bev_camera:
                [B, camera_channels, bev_h, bev_w]
        """

        B, N_cam, _, H, W = images.shape

        intrinsics = intrinsics.to(device=images.device, dtype=images.dtype)
        extrinsics = extrinsics.to(device=images.device, dtype=images.dtype)

        # ResNet + FPN：取 P2（stride=4，分辨率最高，适合做逐像素深度预测）
        # p2: [B*N_cam, camera_channels, H/4, W/4]
        p2, _, _, _ = self.camera_encoder(images)

        _, C, h, w = p2.shape

        # Lift：2D 特征 -> 3D 视锥特征
        # frustum_feature: [B*N_cam, D, C, h, w]
        frustum_feature = self.view_transform(p2)
        D = frustum_feature.shape[1]

        # 还原相机维：[B*N_cam, D, C, h, w] -> [B, N_cam, D, C, h, w]
        frustum_feature = frustum_feature.view(B, N_cam, D, C, h, w)

        # 生成每个 (d, h, w) 采样点的 (u, v, depth)
        frustum = self.frustum_generator(
            h, w,
            device=images.device,
            dtype=images.dtype,
        )  # [D, h, w, 3]

        # 反投影到 LiDAR 坐标系
        points_lidar = self.geometry_transform(
            frustum, intrinsics, extrinsics,
        )  # [B, N_cam, D, h, w, 3]

        # Splat：把视锥特征池化到 BEV 网格
        bev_camera = self.bev_pool(
            frustum_feature, points_lidar,
        )  # [B, C, bev_h, bev_w]

        return bev_camera

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
            self,
            points,
            images,
            intrinsics,
            extrinsics,
            decode=False,
    ):
        """
        Args:
            points:
                [B, N, 4]  LiDAR 点云，[x, y, z, intensity]。

            images:
                [B, N_cam, 3, H, W] 环视图像。

            intrinsics:
                [B, N_cam, 3, 3] 相机内参。

            extrinsics:
                [B, N_cam, 4, 4] 相机 -> LiDAR 外参。

            decode:
                是否顺便跑一遍检测框解码（推理时用，训练时不需要）。

        Returns:
            dict:
                bev_lidar:   [B, lidar_channels,  bev_h, bev_w]
                bev_camera:  [B, camera_channels, bev_h, bev_w]
                bev_fused:   [B, fusion_channels, bev_h, bev_w]
                predictions: CenterHead 输出的各检测分支
                detections:  仅当 decode=True 时存在，解码出的 3D 框
        """

        # -------- LiDAR 分支 --------
        bev_lidar = self.lidar_encoder(points)
        # [B, lidar_channels, bev_h, bev_w]

        # -------- Camera 分支 --------
        bev_camera = self.encode_camera(images, intrinsics, extrinsics)
        # [B, camera_channels, bev_h, bev_w]

        # -------- 融合 --------
        bev_fused = self.fusion(bev_camera, bev_lidar)
        # [B, fusion_channels, bev_h, bev_w]

        # -------- 检测头 --------
        predictions = self.head(bev_fused)

        outputs = {
            "bev_lidar": bev_lidar,
            "bev_camera": bev_camera,
            "bev_fused": bev_fused,
            "predictions": predictions,
        }

        if decode:
            outputs["detections"] = self.decoder(predictions)

        return outputs

    # ==========================================================
    # 从 Config 构造
    # ==========================================================

    @classmethod
    def from_config(cls, config):
        """
        用 configs/default.py 里的 Config 构造模型。

        Args:
            config: 一个带有模型超参数的配置对象（类或实例都行）。

        Returns:
            BEVFusionModel
        """

        resnet_layers = getattr(config, "resnet_layers", (3, 4, 6, 3))
        resnet_channels = getattr(config, "resnet_channels", (256, 512, 1024, 2048))

        return cls(
            point_cloud_range=tuple(config.point_cloud_range),
            voxel_size=tuple(config.voxel_size),
            max_points_per_pillar=config.max_points_per_pillar,
            max_pillars=config.max_pillars,
            lidar_channels=config.bev_channels,
            resnet_layers=tuple(resnet_layers),
            resnet_channels=tuple(resnet_channels),
            camera_channels=config.camera_channels,
            depth_bins=config.depth_bins,
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            fusion_channels=config.fusion_channels,
            num_classes=config.num_classes,
            hidden_channels=getattr(config, "hidden_channels", 128),
            top_k=getattr(config, "top_k", 100),
            score_threshold=getattr(config, "score_threshold", 0.1),
        )
