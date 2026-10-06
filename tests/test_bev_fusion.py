import os
import sys

import torch

# 允许直接运行: python tests/test_bev_fusion.py
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from models.fusion.bev_fusion import BEVFusion


def main():

    B = 2
    C_lidar = 256
    C_camera = 256
    H = 200
    W = 200

    # -----------------------------
    # 模拟 LiDAR BEV
    # -----------------------------

    lidar_bev = torch.randn(
        B,
        C_lidar,
        H,
        W
    )

    # -----------------------------
    # 模拟 Camera BEV
    # -----------------------------

    camera_bev = torch.randn(
        B,
        C_camera,
        H,
        W
    )

    print("LiDAR BEV:")
    print(lidar_bev.shape)

    print("Camera BEV:")
    print(camera_bev.shape)

    # -----------------------------
    # Fusion
    # -----------------------------

    model = BEVFusion(
        lidar_channels=C_lidar,
        camera_channels=C_camera,
        out_channels=256,
    )

    # 注意参数顺序: forward(bev_camera, bev_lidar)
    fused_bev = model(
        camera_bev,
        lidar_bev
    )

    print("Fused BEV:")
    print(fused_bev.shape)

    assert fused_bev.shape == (B, 256, H, W)
    assert torch.isfinite(fused_bev).all()

    # 融合结果必须同时依赖两路输入：
    # 只改 LiDAR 输入，输出应该发生变化。
    perturbed = model(
        camera_bev,
        lidar_bev + 1.0
    )

    diff = (fused_bev - perturbed).abs().max().item()

    print("只改 LiDAR 输入，输出最大变化 =", diff)

    assert diff > 0

    print("=" * 50)
    print("BEVFusion test PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()