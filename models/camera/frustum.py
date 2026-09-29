import torch
import torch.nn as nn


class FrustumGenerator(nn.Module):
    """
    生成视锥采样点坐标。

    为特征图上的每个像素 (h, w) 和每个深度 bin d
    生成一个采样点 (u, v, depth)

        u, v  : 该像素对应到原图上的像素坐标
        depth : 该深度 bin 对应的真实深度值

    输出形状: [D, H, W, 3]，最后一维为 (u, v, depth)。
    后续 geometry 会用它反投影出 3D 坐标。
    """

    def __init__(
        self,
        depth_bins=80,
        depth_min=1.0,
        depth_max=80.0,
        stride=4,
    ):
        """
        Args:
            depth_bins: 深度 bin 数量
            depth_min:  最小深度 (m)
            depth_max:  最大深度 (m)
            stride:     feature map 相对原图的下采样倍数
        """
        super().__init__()

        self.depth_bins = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.stride = stride

    def forward(self, H, W, device=None, dtype=torch.float32):
        """
        Args:
            H, W: feature map 的高宽

        Returns:
            frustum: [D, H, W, 3]，最后一维为 (u, v, depth)
        """

        # 深度 bin：[D]，在 [depth_min, depth_max] 上均匀取值
        depths = torch.linspace(
            self.depth_min, self.depth_max, self.depth_bins,
            device=device, dtype=dtype,
        )

        # feature map 上的整数坐标
        xs = torch.arange(W, device=device, dtype=dtype) # [H]
        ys = torch.arange(H, device=device, dtype=dtype) # [W]

        # meshgrid -> [H, W]
        #    yy[h, w] = h，xx[h, w] = w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij") # xx yy: [H, W]

        # feature 坐标 -> 原图像素坐标
        # 加 0.5 表示取每个特征格子的中心
        # 例如 stride=4，feature x=0 -> 原图 x=(0+0.5)*4=2
        xx = (xx + 0.5) * self.stride
        yy = (yy + 0.5) * self.stride

        # 扩展到 [D, H, W]
        xx = xx.unsqueeze(0).expand(self.depth_bins, -1, -1)
        yy = yy.unsqueeze(0).expand(self.depth_bins, -1, -1)
        depths = depths.view(self.depth_bins, 1, 1).expand(-1, H, W)

        # 拼成 [D, H, W, 3]，最后一维 (u, v, depth)
        frustum = torch.stack([xx, yy, depths], dim=-1)

        return frustum