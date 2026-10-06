# BEVFusion 实现

一个用纯 PyTorch 手写的 BEVFusion 精简实现，目标是**把多模态 3D 检测的每一步摊开写清楚**，
而不是复现论文的精度。

- 只依赖 `torch` / `torchvision` / `numpy`，纯 CPU 就能跑通
- 每个模块单独成文件，输入输出形状都写在 docstring 里
- `tests/` 下的测试不只是断言形状，还会验证**数值正确性**（几何反投影、BEV 散射位置、梯度回传）
- 没有训练脚本、没有数据集加载——重点是把模型结构串起来

---

## 一、整体数据流

```
                    ┌──────────────────── LiDAR 分支 ────────────────────┐
 points [B,N,4]     │  Pillarizer → PillarFeatureBuilder → PFNLayer      │
                    │             → Scatter                              │
                    └───────────────────────┬───────────────────────────┘
                                            │  bev_lidar [B, C_l, H, W]
                                            ▼
                                    ┌───────────────┐
                                    │   BEVFusion   │  concat + 1x1 conv
                                    └───────┬───────┘
                                            │  bev_fused [B, C_f, H, W]
                                            ▼
                                      ┌───────────┐
                                      │ CenterHead│
                                      └─────┬─────┘
                                            │
                                            ▼
                      heatmap / offset / height / size / rotation / velocity
                                            │
                                            ▼
                                  CenterPointDecoder
                                            │
                                            ▼
                                   3D Box [B, K, 9]

                    ┌──────────────────── Camera 分支 ───────────────────┐
 images             │  CameraEncoder (ResNet + FPN)  → P2                │
 [B,N_cam,3,H,W]    │            ↓                                       │
                    │  ViewTransform  (Lift)  2D → 3D 视锥特征            │
                    │            ↓                                       │
                    │  FrustumGenerator  (u, v, depth) 采样网格           │
                    │            ↓                                       │
                    │  GeometryTransform  反投影到 LiDAR 坐标系           │
                    │            ↓                                       │
                    │  BEVPool  (Splat)  视锥特征 → BEV 网格              │
                    └───────────────────────┬───────────────────────────┘
                                            │  bev_camera [B, C_c, H, W]
                                            └──────────► 送入 BEVFusion
```

一句话概括：**LiDAR 分支**把点云压成俯视图特征，**Camera 分支**用 Lift-Splat-Shoot 把图像
也抬成俯视图特征，两路在同一个 BEV 栅格上拼起来，最后交给 CenterPoint 风格的检测头出框。

---

## 二、目录结构

```
bevfusion/
├── configs/
│   └── default.py               # 所有超参数集中在这里
│
├── models/
│   ├── bevfusion.py             # ★ 总模型 BEVFusionModel，把下面全部串起来
│   │
│   ├── lidar/                   # ---------- LiDAR 分支 ----------
│   │   ├── pillarizer.py        # 点云 → Pillar，输出 pillars + coords
│   │   ├── pillar_feature.py    # 4 维点特征 → 9 维（补相对偏移）
│   │   ├── pfn.py               # PFNLayer：逐点 MLP + pillar 内 max pooling
│   │   ├── scatter.py           # pillar 特征散射回 BEV 特征图
│   │   └── lidar_encoder.py     # 上面四步串成 LidarEncoder
│   │
│   ├── camera/                  # ---------- Camera 分支 ----------
│   │   ├── backbone/
│   │   │   ├── bottleneck.py    # ResNet 的 Bottleneck 残差块
│   │   │   ├── resnet.py        # 输出 C2~C5 四个尺度
│   │   │   └── fpn.py           # 自顶向下融合，输出 P2~P5
│   │   ├── camera_encoder.py    # ResNet + FPN，输出多尺度 2D 特征
│   │   ├── depth_net.py         # 每个像素预测 depth_bins 个深度 logits
│   │   ├── view_transform.py    # Lift：深度分布 × 特征 → 5D 视锥特征
│   │   ├── frustum.py           # 生成 (u, v, depth) 采样网格
│   │   ├── geometry.py          # 反投影：像素 + 深度 + 内外参 → LiDAR 坐标
│   │   └── bev_pool.py          # Splat：视锥特征池化到 BEV 网格
│   │
│   ├── fusion/
│   │   └── bev_fusion.py        # 通道维 concat + 1x1 卷积融合
│   │
│   └── heads/                   # ---------- 检测头 ----------
│       ├── detection_head.py    # CenterHead：6 个 dense 预测分支
│       ├── target.py            # GT → Gaussian heatmap + 回归 target
│       ├── loss.py              # Gaussian Focal Loss + masked L1
│       └── decoder.py           # heatmap top-k → 3D Box
│
├── tests/
│   ├── test_pillarizer.py
│   ├── test_lidar_encoder.py
│   ├── test_bev_fusion.py
│   ├── test_detection_head.py
│   └── test_full_pipeline.py    # ★ 完整链路测试，建议先看这个
│
└── requirements.txt
```

---

## 三、张量形状速查

默认配置（`configs/default.py`）：

| 参数 | 值 |
| --- | --- |
| `point_cloud_range` | `[0, -20, -3, 40, 20, 1]` (m) |
| `voxel_size` | `[0.5, 0.5, 4.0]` (m) |
| `bev_h × bev_w` | `(20-(-20))/0.5 × (40-0)/0.5 = 80 × 80` |
| `depth_bins` | 80 |
| `bev_channels` (LiDAR) | 64 |
| `camera_channels` | 64 |
| `fusion_channels` | 128 |
| `num_classes` | 3 |

以 `B=1, N_cam=1, image=256×512` 为例：

| 阶段 | 张量 | 形状 |
| --- | --- | --- |
| 输入 | `points` | `[1, N, 4]` |
| | `images` | `[1, 1, 3, 256, 512]` |
| Pillarizer | `pillars` / `coords` | `[N_total, 32, 4]` / `[N_total, 3]` |
| PillarFeatureBuilder | `pillar_features` | `[N_total, 32, 9]` |
| PFNLayer | `pfn_features` | `[N_total, 64]` |
| Scatter | `bev_lidar` | `[1, 64, 80, 80]` |
| CameraEncoder | `p2` | `[1, 64, 64, 128]` |
| ViewTransform | `frustum_feature` | `[1, 80, 64, 64, 128]` |
| FrustumGenerator | `frustum` | `[80, 64, 128, 3]` |
| GeometryTransform | `points_lidar` | `[1, 1, 80, 64, 128, 3]` |
| BEVPool | `bev_camera` | `[1, 64, 80, 80]` |
| BEVFusion | `bev_fused` | `[1, 128, 80, 80]` |
| CenterHead | `heatmap` | `[1, 3, 80, 80]` |

> `coords` 的最后一维是 `[batch_id, pillar_y, pillar_x]`，不是 `(x, y)`，别搞反。

---

## 四、快速开始

```bash
pip install -r requirements.txt

# 完整链路测试（推荐先跑这个）
python tests/test_full_pipeline.py

# 单个模块的测试
python tests/test_pillarizer.py
python tests/test_lidar_encoder.py
python tests/test_bev_fusion.py
python tests/test_detection_head.py
```

### 最小调用示例

```python
import torch
from models import BEVFusionModel
from configs.default import Config

model = BEVFusionModel.from_config(Config)
model.eval()

B, N_cam = 1, 1

points = torch.rand(B, 20000, 4)            # [x, y, z, intensity]
images = torch.randn(B, N_cam, 3, 256, 512)

intrinsics = torch.eye(3).view(1, 1, 3, 3)  # [B, N_cam, 3, 3]
intrinsics[..., 0, 0] = 250.0               # fx
intrinsics[..., 1, 1] = 250.0               # fy
intrinsics[..., 0, 2] = 256.0               # cx
intrinsics[..., 1, 2] = 128.0               # cy

extrinsics = torch.eye(4).view(1, 1, 4, 4)  # [B, N_cam, 4, 4] 相机 -> LiDAR

with torch.no_grad():
    outputs = model(points, images, intrinsics, extrinsics, decode=True)

print(outputs["bev_lidar"].shape)            # [1, 64, 80, 80]
print(outputs["bev_camera"].shape)           # [1, 64, 80, 80]
print(outputs["bev_fused"].shape)            # [1, 128, 80, 80]
print(outputs["detections"]["boxes"].shape)  # [1, 100, 9]
```

### 训练时的 target / loss

总模型的 `forward` 只做前向推理，损失在外部组合：

```python
from models.heads import CenterPointTargetGenerator, CenterPointLoss

target_gen = CenterPointTargetGenerator(
    voxel_size=(0.5, 0.5),
    point_cloud_range=(0.0, -20.0),
)
criterion = CenterPointLoss()

outputs = model(points, images, intrinsics, extrinsics)
targets = target_gen(gt_boxes, gt_labels, B, num_classes=3, bev_h=80, bev_w=80)

losses = criterion(outputs["predictions"], targets)
losses["loss"].backward()
```

完整写法见 `tests/test_full_pipeline.py` 里的 `test_one_training_step()`。

---

## 五、几个关键概念

### 1. BEV 栅格约定

所有分支共用同一个栅格：

```
iy = floor((y - y_min) / dy)      # 行，对应 H
ix = floor((x - x_min) / dx)      # 列，对应 W
特征图下标: [B, C, iy, ix]
```

**H 对应 y，W 对应 x**。写反了形状照样对，但结果会转 90°，测试里专门有
`test_bev_pool_scatter` 来卡住这个位置关系。

### 2. 外参方向

`GeometryTransform` 按**行向量**约定做矩阵乘：

```
p_lidar = p_cam · Eᵀ       等价于       p_lidar = R @ p_cam + t
```

所以传入的 `extrinsics` 是 **相机 → LiDAR** 的变换矩阵 `T_lidar_cam`。

### 3. Lift-Splat-Shoot 三步

| 步骤 | 模块 | 做什么 |
| --- | --- | --- |
| **Lift** | `DepthNet` + `ViewTransform` | 每个像素预测 `depth_bins` 个深度的概率分布，用它把 2D 特征广播成 3D 视锥特征 `[B*N_cam, D, C, h, w]` |
| **Splat** | `FrustumGenerator` + `GeometryTransform` + `BEVPool` | 为每个 `(d, h, w)` 算出 `(u, v, depth)`，反投影成 LiDAR 坐标，再把对应特征累加到 BEV 栅格上 |
| **Shoot** | `CenterHead` | 在 BEV 上做逐像素的类别 + 框回归（就是 CenterPoint） |

`BEVPool` 用的是 **累加**（`index_add_`）而不是求平均——同一个 BEV 格子会被多个深度上的
视锥点打中，累加相当于把不同深度的证据叠加起来。后面 `BEVFusion` 里的 BatchNorm 会负责
把尺度拉回来。

### 4. 两个分支怎么对齐

LiDAR 分支和 Camera 分支的 BEV 尺寸、栅格分辨率、坐标系完全一致，所以 `BEVFusion`
可以直接在通道维 `concat`（`[B,64,80,80]` + `[B,64,80,80]` → `[B,128,80,80]`），
再用 1×1 卷积压到 `fusion_channels`。

---

## 六、各部分在代码里的位置

| 想搞清楚的问题 | 看哪里 |
| --- | --- |
| 点云怎么变成 pillar 的？ | `models/lidar/pillarizer.py` |
| 9 维点特征是怎么来的？ | `models/lidar/pillar_feature.py` |
| PFN 怎么做 max pooling？ | `models/lidar/pfn.py` |
| 散乱的 pillar 怎么变回规则 BEV？ | `models/lidar/scatter.py` |
| ResNet / FPN 结构 | `models/camera/backbone/` |
| 深度概率怎么预测？ | `models/camera/depth_net.py` |
| 2D 特征怎么抬成 3D？ | `models/camera/view_transform.py` |
| 视锥采样点怎么生成？ | `models/camera/frustum.py` |
| 像素怎么反投影到 LiDAR？ | `models/camera/geometry.py` |
| 视图变换怎么池化到 BEV？ | `models/camera/bev_pool.py` |
| 两路特征怎么融合？ | `models/fusion/bev_fusion.py` |
| 检测头输出什么？ | `models/heads/detection_head.py` |
| GT 怎么变成 heatmap？ | `models/heads/target.py` |
| 损失怎么算？ | `models/heads/loss.py` |
| top-k 怎么解码成框？ | `models/heads/decoder.py` |
| 全部怎么串起来？ | `models/bevfusion.py` |

---

## 七、这个实现省略了什么

这是教学版本，为了把主干讲清楚，刻意省掉了真实系统里的很多东西：

- **没有数据集和训练流程**（nuScenes / Waymo 的加载、数据增强、分布式训练）
- **没有体素化**，LiDAR 分支走的是 PointPillars，不是 VoxelNet
- **没有 BEV 卷积主干**（真实 BEVFusion 在融合后会接一串残差块）
- **单尺度相机特征**，只用了 FPN 的 P2，没有多尺度融合
- **没有深度监督**，`DepthNet` 的深度分布是纯隐式学出来的
- `Scatter` / `BEVPool` 里有 Python 循环，可读性优先，不是性能最优写法

---

## 八、参考

- [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542) (ICRA 2023)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784) (CVPR 2019)
- [Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs](https://arxiv.org/abs/2008.05711) (ECCV 2020)
- [CenterPoint: Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (CVPR 2021)
