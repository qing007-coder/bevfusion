import os
import sys

import torch

# 允许直接运行: python tests/test_pillarizer.py
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from models.lidar.pillarizer import Pillarizer


def main():
    pillarizer = Pillarizer(
        point_cloud_range=[
            0.0, -20.0, -3.0,
            40.0, 20.0, 1.0
        ],

        voxel_size=[
            0.5, 0.5, 4.0
        ],

        max_points_per_pillar=32,

        max_pillars=12000
    )

    print("BEV Spatial Shape:")
    print(pillarizer.spatial_shape)

    print()


    B = 2
    N = 1000

    points = torch.zeros(
        B,
        N,
        4
    )

    points[0, :, 0] = torch.rand(N) * 40.0
    points[0, :, 1] = torch.rand(N) * 40.0 - 20.0
    points[0, :, 2] = torch.rand(N) * 4.0 - 3.0
    points[0, :, 3] = torch.rand(N)


    points[1, :, 0] = torch.rand(N) * 40.0
    points[1, :, 1] = torch.rand(N) * 40.0 - 20.0
    points[1, :, 2] = torch.rand(N) * 4.0 - 3.0
    points[1, :, 3] = torch.rand(N)



    points[0, 0] = torch.tensor([
        -10.0,
        0.0,
        0.0,
        1.0
    ])

    points[0, 1] = torch.tensor([
        50.0,
        0.0,
        0.0,
        1.0
    ])

    points[1, 0] = torch.tensor([
        10.0,
        30.0,
        0.0,
        1.0
    ])


    pillars, coords = pillarizer(points)

    print("Input:")
    print("points.shape =", points.shape)

    print()

    print("Output:")
    print("pillars.shape =", pillars.shape)
    print("coords.shape  =", coords.shape)

    print()



    batch_ids = coords[:, 0]

    for batch_id in range(B):

        num_pillars = (
            batch_ids == batch_id
        ).sum().item()

        print(
            f"Batch {batch_id}: "
            f"{num_pillars} pillars"
        )

    print()

    print("First 10 pillar coords:")

    print(
        coords[:10]
    )

    print()


    assert pillars.shape[0] == coords.shape[0]

    assert pillars.shape[1] == 32

    assert pillars.shape[2] == 4

    assert coords.shape[1] == 3

    assert (
        coords[:, 0].min() >= 0
        and
        coords[:, 0].max() < B
    )

    assert (
        coords[:, 1].min() >= 0
        and
        coords[:, 1].max() < pillarizer.bev_h
    )

    assert (
        coords[:, 2].min() >= 0
        and
        coords[:, 2].max() < pillarizer.bev_w
    )

    # 一个 Pillar 中 32 个位置
    # 全部为 0 的位置代表 padding

    non_zero_points = (
        pillars.abs().sum(dim=2) > 0
    )

    print(
        "Non-zero point slots:",
        non_zero_points.sum().item()
    )

    print()


    print("=" * 50)

    print("Pillarizer test PASSED!")

    print("=" * 50)


if __name__ == "__main__":
    main()