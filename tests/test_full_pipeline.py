"""
BEVFusion 完整链路测试。

覆盖从原始输入到 3D 检测框的整条通路：

    points  -> LidarEncoder  ------------------------\
                                                      -> BEVFusion -> CenterHead -> 解码
    images  -> CameraEncoder -> ViewTransform ->      /
               FrustumGenerator -> GeometryTransform
               -> BEVPool ---------------------------/

除了形状，本文件还验证：
    1. 总模型 forward 与「手动一步步串起来」的结果一致；
    2. 相机几何反投影的数值正确（内参 / 外参 / 深度）；
    3. BEVPool 把特征撒到了正确的 BEV 栅格；
    4. 样本之间互不干扰；
    5. target + loss + backward 能跑通，梯度能回传到两个分支；
    6. 全部点云在范围外时模型不崩、输出全 0；
    7. 能用 configs/default.py 的 Config 构造模型。

运行方式：

    python tests/test_full_pipeline.py

注意：
    部分用例把模型切到了 eval()，因为点云为空时 BatchNorm 在 train
    模式下会因为没有样本可以统计而报错。
"""

import os
import sys

import torch

# 允许直接运行: python tests/test_full_pipeline.py
# 把仓库根目录加进 sys.path，否则找不到 models / configs 包。
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from configs.default import Config
from models import BEVFusionModel
from models.camera.bev_pool import BEVPool
from models.camera.frustum import FrustumGenerator
from models.camera.geometry import GeometryTransform
from models.heads import (
    CenterPointDecoder,
    CenterPointLoss,
    CenterPointTargetGenerator,
)

# ============================================================
# 测试用的小配置
# ============================================================

# 和 configs/default.py 保持一致的空间范围 / 栅格分辨率
# bev_h = (20 - (-20)) / 0.5 = 80
# bev_w = (40 - 0)    / 0.5 = 80
POINT_CLOUD_RANGE = (0.0, -20.0, -3.0, 40.0, 20.0, 1.0)
VOXEL_SIZE = (0.5, 0.5, 4.0)
BEV_SHAPE = (80, 80)

# 为了测试跑得快，把网络缩得很小：
# 真配置是 ResNet-50 [3,4,6,3] + 80 个 depth bin，CPU 上跑一次要好几秒。
TINY_MODEL_KWARGS = dict(
    point_cloud_range=POINT_CLOUD_RANGE,
    voxel_size=VOXEL_SIZE,
    max_points_per_pillar=16,
    max_pillars=2000,
    lidar_channels=16,
    resnet_layers=(1, 1, 1, 1),
    resnet_channels=(256, 512, 1024, 2048),
    camera_channels=16,
    depth_bins=8,
    depth_min=1.0,
    depth_max=20.0,
    fusion_channels=32,
    num_classes=3,
    hidden_channels=32,
    top_k=20,
    score_threshold=0.1,
)

IMAGE_H = 64
IMAGE_W = 64

EXPECTED_CAMERA_FEATURE_HW = (IMAGE_H // 4, IMAGE_W // 4)  # P2 下采样 4 倍


def build_model(**kwargs):
    """构造一个小号 BEVFusionModel，允许覆盖任意参数。"""

    params = dict(TINY_MODEL_KWARGS)
    params.update(kwargs)

    return BEVFusionModel(**params)


def make_inputs(
        batch_size=2,
        num_points=500,
        num_cameras=2,
        seed=0,
        fx=60.0,
        fy=60.0,
):
    """
    构造一组合法的模型输入。

    Returns:
        points:     [B, N, 4]
        images:     [B, N_cam, 3, H, W]
        intrinsics: [B, N_cam, 3, 3]
        extrinsics: [B, N_cam, 4, 4]  单位阵，即相机系 == LiDAR 系
    """

    generator = torch.Generator().manual_seed(seed)

    # ---- 点云：均匀落在 point_cloud_range 内 ----
    x = torch.rand(batch_size, num_points, 1, generator=generator)
    y = torch.rand(batch_size, num_points, 1, generator=generator)
    z = torch.rand(batch_size, num_points, 1, generator=generator)
    intensity = torch.rand(batch_size, num_points, 1, generator=generator)

    x = x * (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) + POINT_CLOUD_RANGE[0]
    y = y * (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) + POINT_CLOUD_RANGE[1]
    z = z * (POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) + POINT_CLOUD_RANGE[2]

    points = torch.cat([x, y, z, intensity], dim=2)
    # [B, N, 4]

    # ---- 图像 ----
    images = torch.randn(
        batch_size, num_cameras, 3, IMAGE_H, IMAGE_W,
        generator=generator,
    )

    # ---- 内参：主点放在图像中心 ----
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(
        batch_size, num_cameras, 1, 1
    )

    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = IMAGE_W / 2.0
    intrinsics[..., 1, 2] = IMAGE_H / 2.0

    # ---- 外参：相机 -> LiDAR，这里取单位阵 ----
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch_size, num_cameras, 1, 1
    )

    return points, images, intrinsics, extrinsics


def make_gt(batch_size=2, num_boxes=2, seed=0):
    """
    构造 GT 框，[x, y, z, w, l, h, yaw, vx, vy]，全部落在 BEV 范围内。
    """

    generator = torch.Generator().manual_seed(seed)

    gt_boxes = torch.zeros(batch_size, num_boxes, 9)
    gt_boxes[..., 0] = 12.0 + torch.rand(batch_size, num_boxes, generator=generator) * 10.0
    gt_boxes[..., 1] = -8.0 + torch.rand(batch_size, num_boxes, generator=generator) * 16.0
    gt_boxes[..., 2] = -1.0
    gt_boxes[..., 3] = 1.8    # w
    gt_boxes[..., 4] = 4.2    # l
    gt_boxes[..., 5] = 1.5    # h
    gt_boxes[..., 6] = 0.3    # yaw
    gt_boxes[..., 7] = 2.0    # vx
    gt_boxes[..., 8] = 0.0    # vy

    gt_labels = torch.zeros(batch_size, num_boxes, dtype=torch.long)
    gt_labels[:, 1] = 1

    return gt_boxes, gt_labels


# ============================================================
# 1. forward 形状
# ============================================================

def test_forward_shapes():

    print("=" * 60)
    print("test_forward_shapes")
    print("=" * 60)

    model = build_model()
    model.eval()

    B, N_cam = 2, 2

    points, images, intrinsics, extrinsics = make_inputs(
        batch_size=B, num_cameras=N_cam,
    )

    with torch.no_grad():
        outputs = model(points, images, intrinsics, extrinsics)

    bev_lidar = outputs["bev_lidar"]
    bev_camera = outputs["bev_camera"]
    bev_fused = outputs["bev_fused"]
    predictions = outputs["predictions"]

    print("points        =", tuple(points.shape))
    print("bev_lidar     =", tuple(bev_lidar.shape))
    print("bev_camera    =", tuple(bev_camera.shape))
    print("bev_fused     =", tuple(bev_fused.shape))

    for name, value in predictions.items():
        print(f"  pred {name:9s} =", tuple(value.shape))

    # LiDAR 分支
    assert bev_lidar.shape == (B, 16, BEV_SHAPE[0], BEV_SHAPE[1])

    # Camera 分支
    assert bev_camera.shape == (B, 16, BEV_SHAPE[0], BEV_SHAPE[1])

    # 融合后
    assert bev_fused.shape == (B, 32, BEV_SHAPE[0], BEV_SHAPE[1])

    # 检测头：6 个分支的通道数分别是 C / 2 / 1 / 3 / 2 / 2
    assert predictions["heatmap"].shape == (B, 3, BEV_SHAPE[0], BEV_SHAPE[1])
    assert predictions["offset"].shape == (B, 2, BEV_SHAPE[0], BEV_SHAPE[1])
    assert predictions["height"].shape == (B, 1, BEV_SHAPE[0], BEV_SHAPE[1])
    assert predictions["size"].shape == (B, 3, BEV_SHAPE[0], BEV_SHAPE[1])
    assert predictions["rotation"].shape == (B, 2, BEV_SHAPE[0], BEV_SHAPE[1])
    assert predictions["velocity"].shape == (B, 2, BEV_SHAPE[0], BEV_SHAPE[1])

    # 两个分支都必须真的产生了非零特征，
    # 否则说明视锥点全部落在 BEV 之外 / 点云被全部过滤。
    print("bev_lidar.abs().sum()  =", bev_lidar.abs().sum().item())
    print("bev_camera.abs().sum() =", bev_camera.abs().sum().item())

    assert bev_lidar.abs().sum() > 0
    assert bev_camera.abs().sum() > 0

    assert torch.isfinite(bev_fused).all()

    print()

    return model


# ============================================================
# 2. 手动串一遍，结果必须和 forward 一致
# ============================================================

def test_manual_pipeline_matches_forward():

    print("=" * 60)
    print("test_manual_pipeline_matches_forward")
    print("=" * 60)

    model = build_model()
    model.eval()

    B, N_cam = 2, 2

    points, images, intrinsics, extrinsics = make_inputs(
        batch_size=B, num_cameras=N_cam,
    )

    with torch.no_grad():

        outputs = model(points, images, intrinsics, extrinsics)

        # ---- 手动把每一步拆开 ----
        bev_lidar = model.lidar_encoder(points)
        bev_camera = model.encode_camera(images, intrinsics, extrinsics)
        bev_fused = model.fusion(bev_camera, bev_lidar)
        predictions = model.head(bev_fused)

    for name, manual in [
        ("bev_lidar", bev_lidar),
        ("bev_camera", bev_camera),
        ("bev_fused", bev_fused),
    ]:
        diff = (outputs[name] - manual).abs().max().item()
        print(f"{name:12s} max diff = {diff}")
        assert torch.allclose(outputs[name], manual, atol=1e-6)

    for name, value in predictions.items():
        diff = (outputs["predictions"][name] - value).abs().max().item()
        print(f"pred {name:9s} max diff = {diff}")
        assert torch.allclose(outputs["predictions"][name], value, atol=1e-6)

    print()


# ============================================================
# 3. 相机几何：反投影的数值必须正确
# ============================================================

def test_geometry_transform_numerics():

    print("=" * 60)
    print("test_geometry_transform_numerics")
    print("=" * 60)

    D, H, W = 4, 5, 6
    stride = 4
    fx, fy = 60.0, 50.0
    cx, cy = 30.0, 20.0
    depth_min, depth_max = 2.0, 10.0

    frustum_gen = FrustumGenerator(
        depth_bins=D,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
    )

    frustum = frustum_gen(H, W)  # [D, H, W, 3] -> (u, v, depth)
    print("frustum           =", tuple(frustum.shape))

    assert frustum.shape == (D, H, W, 3)

    # 每个 (d, h, w) 的 depth 必须是均匀分布的深度 bin
    expected_depths = torch.linspace(depth_min, depth_max, D)
    assert torch.allclose(frustum[..., 2], expected_depths.view(D, 1, 1).expand(D, H, W))

    # 特征格子的中心映射回原图像素：(h + 0.5) * stride
    # frustum 最后一维是 (u, v, depth)
    assert frustum[0, 0, 0, 0].item() == 0.5 * stride     # u at (h=0, w=0)
    assert frustum[0, 0, 0, 1].item() == 0.5 * stride     # v at (h=0, w=0)
    assert frustum[0, 0, 1, 0].item() == 1.5 * stride     # u at (h=0, w=1)
    assert frustum[0, 1, 0, 1].item() == 1.5 * stride     # v at (h=1, w=0)

    intrinsics = torch.tensor([[
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]]).unsqueeze(0)  # [1, 1, 3, 3]

    # ---- 情况一：外参为单位阵，相机系 == LiDAR 系 ----
    extrinsics = torch.eye(4).view(1, 1, 4, 4)

    geometry = GeometryTransform()

    points_lidar = geometry(frustum, intrinsics, extrinsics)
    print("points_lidar      =", tuple(points_lidar.shape))

    assert points_lidar.shape == (1, 1, D, H, W, 3)

    u, v, depth = frustum[..., 0], frustum[..., 1], frustum[..., 2]

    expected_x = (u - cx) / fx * depth
    expected_y = (v - cy) / fy * depth
    expected_z = depth

    print("x max diff =", (points_lidar[0, 0, ..., 0] - expected_x).abs().max().item())
    print("y max diff =", (points_lidar[0, 0, ..., 1] - expected_y).abs().max().item())
    print("z max diff =", (points_lidar[0, 0, ..., 2] - expected_z).abs().max().item())

    assert torch.allclose(points_lidar[0, 0, ..., 0], expected_x, atol=1e-4)
    assert torch.allclose(points_lidar[0, 0, ..., 1], expected_y, atol=1e-4)
    assert torch.allclose(points_lidar[0, 0, ..., 2], expected_z, atol=1e-4)

    # ---- 情况二：外参加一个平移 t，p_lidar = R @ p_cam + t ----
    t = torch.tensor([1.0, 2.0, 3.0])

    extrinsics_shifted = torch.eye(4).view(1, 1, 4, 4).clone()
    extrinsics_shifted[..., :3, 3] = t

    points_shifted = geometry(frustum, intrinsics, extrinsics_shifted)

    assert torch.allclose(
        points_shifted[0, 0, ..., 0],
        expected_x + t[0],
        atol=1e-4,
    )
    assert torch.allclose(
        points_shifted[0, 0, ..., 1],
        expected_y + t[1],
        atol=1e-4,
    )
    assert torch.allclose(
        points_shifted[0, 0, ..., 2],
        expected_z + t[2],
        atol=1e-4,
    )

    print("translation 一致")

    # ---- 情况三：纯旋转 90 度 (绕 z 轴) ----
    # R = [[0,-1,0],[1,0,0],[0,0,1]]，则 (x, y, z) -> (-y, x, z)
    rotation = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    extrinsics_rot = torch.eye(4).view(1, 1, 4, 4).clone()
    extrinsics_rot[..., :3, :3] = rotation

    points_rot = geometry(frustum, intrinsics, extrinsics_rot)

    assert torch.allclose(points_rot[0, 0, ..., 0], -expected_y, atol=1e-4)
    assert torch.allclose(points_rot[0, 0, ..., 1], expected_x, atol=1e-4)
    assert torch.allclose(points_rot[0, 0, ..., 2], expected_z, atol=1e-4)

    print("rotation 一致")

    # ---- 情况四：float64 输入也要能用（模型里会统一转成 float32）----
    frustum = frustum_gen(H, W, dtype=torch.float32)
    _ = geometry(frustum, intrinsics, extrinsics)

    print()


# ============================================================
# 4. BEVPool：特征必须落到正确的 BEV 栅格
# ============================================================

def test_bev_pool_scatter():

    print("=" * 60)
    print("test_bev_pool_scatter")
    print("=" * 60)

    B, N_cam, D, C, H, W = 1, 1, 2, 3, 2, 2

    pool = BEVPool(
        x_range=(0.0, 40.0),
        y_range=(-20.0, 20.0),
        voxel_size=(0.5, 0.5),
    )

    assert (pool.bev_h, pool.bev_w) == BEV_SHAPE

    # 视锥特征：全部置 0，只在 (n_cam=0, d=1, h=1, w=0) 放一个 1
    frustum_feature = torch.zeros(B, N_cam, D, C, H, W)
    frustum_feature[0, 0, 1, :, 1, 0] = 1.0

    # 所有点的坐标都放到 BEV 之外，再把目标点改到坐标 (x=1.2, y=-19.3)
    #   ix = floor((1.2 - 0.0) / 0.5)      = 2
    #   iy = floor((-19.3 + 20.0) / 0.5)   = 1
    points_lidar = torch.zeros(B, N_cam, D, H, W, 3)
    points_lidar[0, 0, 1, 1, 0, 0] = 1.2
    points_lidar[0, 0, 1, 1, 0, 1] = -19.3
    points_lidar[0, 0, 1, 1, 0, 2] = 0.0

    bev = pool(frustum_feature, points_lidar)
    print("bev_feature =", tuple(bev.shape))

    assert bev.shape == (B, C, BEV_SHAPE[0], BEV_SHAPE[1])

    occupied = (bev.abs().sum(dim=1) > 0)
    print("occupied cells =", occupied.sum().item())

    # 只有 (iy=1, ix=2) 这一个栅格被写入
    assert occupied.sum().item() == 1
    assert bool(occupied[0, 1, 2])

    # 那个格子上的 C 个通道都应该等于 1
    assert torch.allclose(bev[0, :, 1, 2], torch.ones(C))

    # 范围外的点必须被丢掉：把坐标推到 x 之外，结果应该是全 0
    points_lidar_out = points_lidar.clone()
    points_lidar_out[..., 0] = 100.0

    bev_out = pool(frustum_feature, points_lidar_out)

    assert (bev_out == 0).all()

    print("范围外点被正确过滤")

    print()


# ============================================================
# 5. 样本之间互不干扰
# ============================================================

def test_batch_independence():

    print("=" * 60)
    print("test_batch_independence")
    print("=" * 60)

    model = build_model()

    # BatchNorm 在 train 模式下会统计整个 Batch，这里只关心样本隔离性
    model.eval()

    points, images, intrinsics, extrinsics = make_inputs(batch_size=2)

    with torch.no_grad():
        batched = model(points, images, intrinsics, extrinsics)
        single = model(points[0:1], images[0:1], intrinsics[0:1], extrinsics[0:1])

    for name in ["bev_lidar", "bev_camera", "bev_fused"]:
        diff = (batched[name][0] - single[name][0]).abs().max().item()
        print(f"{name:12s} max diff = {diff}")
        assert torch.allclose(batched[name][0], single[name][0], atol=1e-6)

    print()


# ============================================================
# 6. target + loss + backward
# ============================================================

def test_loss_and_backward():

    print("=" * 60)
    print("test_loss_and_backward")
    print("=" * 60)

    model = build_model()

    B = 2

    points, images, intrinsics, extrinsics = make_inputs(batch_size=B)
    gt_boxes, gt_labels = make_gt(batch_size=B)

    outputs = model(points, images, intrinsics, extrinsics)
    predictions = outputs["predictions"]

    target_generator = CenterPointTargetGenerator(
        voxel_size=(VOXEL_SIZE[0], VOXEL_SIZE[1]),
        point_cloud_range=(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1]),
        min_radius=2,
    )

    targets = target_generator(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        batch_size=B,
        num_classes=3,
        bev_h=BEV_SHAPE[0],
        bev_w=BEV_SHAPE[1],
    )

    print("target heatmap  =", tuple(targets["heatmap"].shape))
    print("reg_mask 命中数 =", targets["reg_mask"].sum().item())

    # 每个 GT 都应该在 reg_mask 上打上点
    assert targets["reg_mask"].sum().item() == B * 2
    # heatmap 峰值必须为 1（高斯圆心）
    assert targets["heatmap"].max().item() == 1.0

    criterion = CenterPointLoss()

    losses = criterion(predictions, targets)

    for name, value in losses.items():
        print(f"  {name:16s} = {value.item():.4f}")
        assert torch.isfinite(value)

    loss = losses["loss"]
    loss.backward()

    # 梯度必须能回传到 Camera 分支和 LiDAR 分支的底层参数上
    assert model.lidar_encoder.pfn.linear.weight.grad is not None
    assert model.lidar_encoder.pfn.linear.weight.grad.abs().sum() > 0

    assert model.camera_encoder.backbone.layer1[0].conv1.weight.grad is not None
    assert model.camera_encoder.backbone.layer1[0].conv1.weight.grad.abs().sum() > 0

    assert model.view_transform.depth_net.depth_head.weight.grad.abs().sum() > 0

    assert model.fusion.fusion_conv[0].weight.grad.abs().sum() > 0

    assert model.head.heatmap_head.weight.grad.abs().sum() > 0

    print("两个分支 + 融合 + 检测头的梯度都正常")

    print()


# ============================================================
# 7. 解码 3D 框
# ============================================================

def test_decode():

    print("=" * 60)
    print("test_decode")
    print("=" * 60)

    model = build_model()
    model.eval()

    B = 2

    points, images, intrinsics, extrinsics = make_inputs(batch_size=B)

    with torch.no_grad():
        outputs = model(
            points, images, intrinsics, extrinsics, decode=True,
        )

    detections = outputs["detections"]

    boxes = detections["boxes"]
    scores = detections["scores"]
    classes = detections["classes"]

    print("boxes   =", tuple(boxes.shape))
    print("scores  =", tuple(scores.shape))
    print("classes =", tuple(classes.shape))

    top_k = TINY_MODEL_KWARGS["top_k"]

    assert boxes.shape == (B, top_k, 9)
    assert scores.shape == (B, top_k)
    assert classes.shape == (B, top_k)

    # 分数经过 sigmoid，必须落在 (0, 1)
    assert (scores > 0).all() and (scores < 1).all()

    # 类别必须合法
    assert (classes >= 0).all() and (classes < TINY_MODEL_KWARGS["num_classes"]).all()

    # 前 7 维是 x, y, z, w, l, h, yaw，后两维是 vx, vy
    assert torch.isfinite(boxes).all()

    # yaw 由 atan2(sin, cos) 得到，必然落在 [-pi, pi]
    assert (boxes[..., 6].abs() <= torch.pi + 1e-5).all()

    print("boxes 范围: x[%.2f, %.2f] y[%.2f, %.2f]"
          % (boxes[..., 0].min().item(), boxes[..., 0].max().item(),
             boxes[..., 1].min().item(), boxes[..., 1].max().item()))

    # ---- 单独测一下 decoder 的 topk 逻辑 ----
    # 构造一个只在 (class=1, y=4, x=5) 有高分的 heatmap
    decoder = CenterPointDecoder(
        top_k=3,
        voxel_size=(VOXEL_SIZE[0], VOXEL_SIZE[1]),
        point_cloud_range=(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1]),
        score_threshold=0.1,
    )

    heatmap = torch.zeros(1, 3, BEV_SHAPE[0], BEV_SHAPE[1])
    heatmap[0, 1, 4, 5] = 10.0  # sigmoid 之后约等于 1

    scores_k, classes_k, ys_k, xs_k = decoder.topk(torch.sigmoid(heatmap))

    print("top1: score=%.3f class=%d y=%d x=%d"
          % (scores_k[0, 0].item(), classes_k[0, 0].item(),
             ys_k[0, 0].item(), xs_k[0, 0].item()))

    assert classes_k[0, 0].item() == 1
    assert ys_k[0, 0].item() == 4
    assert xs_k[0, 0].item() == 5

    print()


# ============================================================
# 8. 点云全部在范围外
# ============================================================

def test_points_out_of_range():

    print("=" * 60)
    print("test_points_out_of_range")
    print("=" * 60)

    model = build_model()

    # 空点云 -> 没有 pillar -> BatchNorm 只有在 eval 下才不报错
    model.eval()

    points, images, intrinsics, extrinsics = make_inputs(batch_size=2)

    # 把点云平移到 x_max 之外
    points[:, :, 0] += 1000.0

    with torch.no_grad():
        outputs = model(points, images, intrinsics, extrinsics)

    bev_lidar = outputs["bev_lidar"]
    bev_camera = outputs["bev_camera"]

    print("bev_lidar.abs().max() =", bev_lidar.abs().max().item())

    assert bev_lidar.shape == (2, 16, BEV_SHAPE[0], BEV_SHAPE[1])
    assert (bev_lidar == 0).all()

    # 相机分支不受影响，应该仍然有特征
    assert bev_camera.abs().sum() > 0
    assert torch.isfinite(outputs["bev_fused"]).all()

    print("相机分支不受影响")

    print()


# ============================================================
# 9. 用 Config 构造模型
# ============================================================

def test_from_config():

    print("=" * 60)
    print("test_from_config")
    print("=" * 60)

    # ---- 真实配置：只检查结构，不跑 forward（太重）----
    model = BEVFusionModel.from_config(Config)

    print("bev_h / bev_w        =", model.bev_h, model.bev_w)
    print("lidar_channels       =", model.lidar_channels)
    print("camera_channels      =", model.camera_channels)
    print("fusion_channels      =", model.fusion_channels)
    print("depth_bins           =", model.depth_bins)

    assert (model.bev_h, model.bev_w) == BEV_SHAPE
    assert model.lidar_channels == Config.bev_channels
    assert model.camera_channels == Config.camera_channels
    assert model.fusion_channels == Config.fusion_channels
    assert model.depth_bins == Config.depth_bins

    # ---- 缩小版配置：真的跑一遍 ----
    class TinyConfig(Config):
        image_h = IMAGE_H
        image_w = IMAGE_W
        resnet_layers = [1, 1, 1, 1]
        camera_channels = 16
        bev_channels = 16
        depth_bins = 8
        depth_min = 1.0
        depth_max = 20.0
        fusion_channels = 32
        hidden_channels = 32
        top_k = 10

    tiny = BEVFusionModel.from_config(TinyConfig)
    tiny.eval()

    points, images, intrinsics, extrinsics = make_inputs(batch_size=1, num_cameras=1)

    with torch.no_grad():
        outputs = tiny(points, images, intrinsics, extrinsics)

    print("bev_lidar  =", tuple(outputs["bev_lidar"].shape))
    print("bev_camera =", tuple(outputs["bev_camera"].shape))
    print("bev_fused  =", tuple(outputs["bev_fused"].shape))

    assert outputs["bev_lidar"].shape == (1, 16, BEV_SHAPE[0], BEV_SHAPE[1])
    assert outputs["bev_camera"].shape == (1, 16, BEV_SHAPE[0], BEV_SHAPE[1])
    assert outputs["bev_fused"].shape == (1, 32, BEV_SHAPE[0], BEV_SHAPE[1])

    print()


# ============================================================
# 10. 训练一步（参数必须真的被更新）
# ============================================================

def test_one_training_step():

    print("=" * 60)
    print("test_one_training_step")
    print("=" * 60)

    torch.manual_seed(0)

    model = build_model()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    B = 2

    points, images, intrinsics, extrinsics = make_inputs(batch_size=B)
    gt_boxes, gt_labels = make_gt(batch_size=B)

    target_generator = CenterPointTargetGenerator(
        voxel_size=(VOXEL_SIZE[0], VOXEL_SIZE[1]),
        point_cloud_range=(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1]),
    )

    criterion = CenterPointLoss()

    # 记录优化前的一组参数
    before = model.fusion.fusion_conv[0].weight.detach().clone()

    optimizer.zero_grad()

    outputs = model(points, images, intrinsics, extrinsics)

    targets = target_generator(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        batch_size=B,
        num_classes=3,
        bev_h=BEV_SHAPE[0],
        bev_w=BEV_SHAPE[1],
    )

    losses = criterion(outputs["predictions"], targets)

    losses["loss"].backward()
    optimizer.step()

    after = model.fusion.fusion_conv[0].weight.detach()

    diff = (after - before).abs().max().item()

    print("loss  =", losses["loss"].item())
    print("参数更新量 =", diff)

    assert losses["loss"].item() > 0
    assert diff > 0

    print("一步训练成功")

    print()


def main():

    test_forward_shapes()

    test_manual_pipeline_matches_forward()

    test_geometry_transform_numerics()

    test_bev_pool_scatter()

    test_batch_independence()

    test_loss_and_backward()

    test_decode()

    test_points_out_of_range()

    test_from_config()

    test_one_training_step()

    print("=" * 60)
    print("BEVFusion full pipeline test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
