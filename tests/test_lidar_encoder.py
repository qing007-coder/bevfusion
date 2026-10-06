import os
import sys

import torch

# 允许直接运行: python tests/test_lidar_encoder.py
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from models.lidar.lidar_encoder import LidarEncoder


POINT_CLOUD_RANGE = [
    0.0, -20.0, -3.0,
    40.0, 20.0, 1.0
]

VOXEL_SIZE = [
    0.5, 0.5, 4.0
]

# bev_h = (20 - (-20)) / 0.5 = 80
# bev_w = (40 - 0) / 0.5 = 80
EXPECTED_BEV_SHAPE = (80, 80)


def build_encoder(**kwargs):
    """
    构造 LidarEncoder，

    默认配置与 test_pillarizer.py 保持一致。
    """

    params = dict(
        point_cloud_range=POINT_CLOUD_RANGE,
        voxel_size=VOXEL_SIZE,
        max_points_per_pillar=32,
        max_pillars=12000,
        in_channels=9,
        out_channels=64,
    )

    params.update(kwargs)

    return LidarEncoder(**params)


def make_points(batch_size, num_points, seed=0):
    """
    随机生成 [B, N, 4] 的点云，

    点云落在 point_cloud_range 内部。
    """

    generator = torch.Generator().manual_seed(seed)

    x = torch.rand(
        batch_size, num_points, 1, generator=generator
    ) * (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) + POINT_CLOUD_RANGE[0]

    y = torch.rand(
        batch_size, num_points, 1, generator=generator
    ) * (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) + POINT_CLOUD_RANGE[1]

    z = torch.rand(
        batch_size, num_points, 1, generator=generator
    ) * (POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) + POINT_CLOUD_RANGE[2]

    intensity = torch.rand(
        batch_size, num_points, 1, generator=generator
    )

    return torch.cat(
        [x, y, z, intensity],
        dim=2
    )


def make_pillar_points(pillar_y, pillar_x, num_points=32):
    """
    生成落在指定 Pillar 内部的点，

    Pillar (pillar_y, pillar_x) 的 XY 范围是:
        x: x_min + pillar_x * vx ~ x_min + (pillar_x + 1) * vx
        y: y_min + pillar_y * vy ~ y_min + (pillar_y + 1) * vy

    返回 [num_points, 4]，全部落在该 Pillar 中心附近。
    """

    vx, vy = VOXEL_SIZE[0], VOXEL_SIZE[1]

    center_x = (
        POINT_CLOUD_RANGE[0] + (pillar_x + 0.5) * vx
    )

    center_y = (
        POINT_CLOUD_RANGE[1] + (pillar_y + 0.5) * vy
    )

    points = torch.zeros(
        num_points,
        4
    )

    points[:, 0] = center_x
    points[:, 1] = center_y
    points[:, 2] = 0.0
    points[:, 3] = 1.0

    return points


def test_output_shape():
    """
    输出形状必须是 [B, out_channels, bev_h, bev_w]
    """

    print("=" * 50)
    print("test_output_shape")
    print("=" * 50)

    model = build_encoder()

    print("bev_shape =", model.bev_shape)

    assert model.bev_shape == EXPECTED_BEV_SHAPE

    B, N = 2, 1000

    points = make_points(
        batch_size=B,
        num_points=N
    )

    bev_features = model(points)

    print("points.shape       =", points.shape)
    print("bev_features.shape =", bev_features.shape)

    assert bev_features.shape == (
        B, 64, EXPECTED_BEV_SHAPE[0], EXPECTED_BEV_SHAPE[1]
    )

    assert bev_features.dtype == torch.float32

    assert torch.isfinite(bev_features).all()

    # 换成别的 out_channels / batch size 也要成立
    model = build_encoder(out_channels=32)

    bev_features = model(
        make_points(
            batch_size=3,
            num_points=200,
            seed=1
        )
    )

    print("out_channels=32 ->", bev_features.shape)

    assert bev_features.shape == (
        3, 32, EXPECTED_BEV_SHAPE[0], EXPECTED_BEV_SHAPE[1]
    )

    print()


def test_manual_pipeline_matches_forward():
    """
    forward 必须等价于手动串起

    Pillarizer -> PillarFeatureBuilder -> PFNLayer -> Scatter
    """

    print("=" * 50)
    print("test_manual_pipeline_matches_forward")
    print("=" * 50)

    model = build_encoder()

    # eval 模式避免 BatchNorm 的 running stats 被更新
    model.eval()

    points = make_points(
        batch_size=2,
        num_points=500
    )

    bev_features = model(points)

    pillars, coords = model.pillarizer(points)
    pillar_features = model.pillar_feature_builder(pillars, coords)
    pfn_features = model.pfn(pillar_features)
    manual_bev = model.scatter(points.shape[0], coords, pfn_features)

    print("pillars.shape        =", pillars.shape)
    print("coords.shape         =", coords.shape)
    print("pillar_features.shape=", pillar_features.shape)
    print("pfn_features.shape   =", pfn_features.shape)

    assert pillars.shape == (
        coords.shape[0], 32, 4
    )

    assert coords.shape[1] == 3

    assert pillar_features.shape == (
        coords.shape[0], 32, 9
    )

    # PFN 在 pillar 内部做 max pooling，输出 [N_total, out_channels]
    assert pfn_features.shape == (
        coords.shape[0], 64
    )

    assert torch.allclose(bev_features, manual_bev)

    print("max diff =", (bev_features - manual_bev).abs().max().item())

    print()


def test_scatter_position():
    """
    每一个非空 Pillar 必须散射到 BEV 特征图上

    正确的 (pillar_y, pillar_x) 位置，且其他位置保持为 0。
    """

    print("=" * 50)
    print("test_scatter_position")
    print("=" * 50)

    model = build_encoder()

    model.eval()

    # 样本 0: 只放一个 Pillar
    # x = 1.0 ~ 1.5 -> pillar_x = 2
    # y = -19.0 ~ -18.5 -> pillar_y = 2
    sample_0 = make_pillar_points(
        pillar_y=2,
        pillar_x=2
    )

    # 样本 1: 只放一个 Pillar
    # 打到 BEV 特征图右上角
    # x = 39.0 ~ 39.5 -> pillar_x = 78
    # y = 19.0 ~ 19.5 -> pillar_y = 78
    sample_1 = make_pillar_points(
        pillar_y=78,
        pillar_x=78
    )

    # 若干个空间范围之外的点，必须被过滤掉
    out_of_range = torch.tensor([
        [-1.0, 0.0, 0.0, 1.0], # x < x_min
        [41.0, 0.0, 0.0, 1.0], # x >= x_max
        [10.0, -21.0, 0.0, 1.0], # y < y_min
        [10.0, 21.0, 0.0, 1.0], # y >= y_max
    ])

    points = torch.stack(
        [
            torch.cat([sample_0, out_of_range], dim=0),
            torch.cat([sample_1, out_of_range], dim=0),
        ],
        dim=0
    )

    pillars, coords = model.pillarizer(points)

    print("coords =")
    print(coords)

    # 范围外的点被过滤后，每个样本只剩下 1 个 Pillar
    assert coords.shape[0] == 2

    assert coords[0].tolist() == [0, 2, 2]

    assert coords[1].tolist() == [1, 78, 78]

    bev_features = model(points)

    # [B, H, W] 每个位置是否有特征
    occupied = (
        bev_features.abs().sum(dim=1) > 0
    )

    print("occupied cells :", occupied.sum().item())

    expected = torch.zeros_like(occupied)

    expected[0, 2, 2] = True
    expected[1, 78, 78] = True

    assert torch.equal(occupied, expected)

    print("occupied == expected")

    print()


def test_batch_independence():
    """
    样本之间不能互相影响:

    bev[0] 与单独跑 points[0] 的结果必须一致。
    """

    print("=" * 50)
    print("test_batch_independence")
    print("=" * 50)

    model = build_encoder()

    # BatchNorm 处于 train 模式时会统计整个 Batch 的均值方差，
    # 这里只关心样本之间的隔离性，所以切到 eval 模式。
    model.eval()

    points = make_points(
        batch_size=2,
        num_points=500
    )

    batched = model(points)
    single = model(points[0:1])

    print("batched[0].shape =", batched[0].shape)
    print("single[0].shape  =", single[0].shape)

    assert torch.allclose(
        batched[0],
        single[0],
        atol=1e-6
    )

    print("max diff =", (batched[0] - single[0]).abs().max().item())

    print()


def test_points_out_of_range():
    """
    点云全部在空间范围之外时，

    BEV 特征图必须全是 0。
    """

    print("=" * 50)
    print("test_points_out_of_range")
    print("=" * 50)

    model = build_encoder()

    model.eval()

    points = make_points(
        batch_size=2,
        num_points=100
    )

    # 平移到 x_max 之外
    points[:, :, 0] += 100.0

    bev_features = model(points)

    print("bev_features.shape =", bev_features.shape)
    print("bev_features.abs().max() =", bev_features.abs().max().item())

    assert bev_features.shape == (
        2, 64, EXPECTED_BEV_SHAPE[0], EXPECTED_BEV_SHAPE[1]
    )

    assert (bev_features == 0).all()

    print()


def test_max_pillars():
    """
    max_pillars 限制生效:

    非空 Pillar 数量不能超过 max_pillars。
    """

    print("=" * 50)
    print("test_max_pillars")
    print("=" * 50)

    max_pillars = 8

    model = build_encoder(
        max_pillars=max_pillars
    )

    model.eval()

    # 200 个点随机分布，几乎不会落在同一个 Pillar 里
    points = make_points(
        batch_size=2,
        num_points=200
    )

    _, coords = model.pillarizer(points)

    bev_features = model(points)

    occupied = (
        bev_features.abs().sum(dim=1) > 0
    ).sum().item()

    print("max_pillars          =", max_pillars)
    print("coords.shape[0]      =", coords.shape[0])
    print("occupied cells       =", occupied)

    assert coords.shape[0] <= max_pillars * 2

    assert occupied <= max_pillars * 2

    print()


def test_backward():
    """
    梯度必须能回传到 PFN 的参数上。
    """

    print("=" * 50)
    print("test_backward")
    print("=" * 50)

    model = build_encoder()

    points = make_points(
        batch_size=2,
        num_points=500
    )

    bev_features = model(points)

    loss = bev_features.sum()
    loss.backward()

    print("loss =", loss.item())

    for name, param in model.named_parameters():

        print(
            f"{name:24s}",
            "grad is None" if param.grad is None else "grad ok"
        )

        assert param.grad is not None

        assert torch.isfinite(param.grad).all()

    assert model.pfn.linear.weight.grad.abs().sum() > 0

    print()


def main():

    test_output_shape()

    test_manual_pipeline_matches_forward()

    test_scatter_position()

    test_batch_independence()

    test_points_out_of_range()

    test_max_pillars()

    test_backward()

    print("=" * 50)
    print("LidarEncoder test PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
