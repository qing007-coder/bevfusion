import math

import torch
import torch.nn as nn


class CenterPointTargetGenerator(nn.Module):

    def __init__(
        self,
        voxel_size=(0.5, 0.5),
        point_cloud_range=(-50.0, -50.0),
        min_radius=2,
    ):
        super().__init__()

        self.voxel_size_x = voxel_size[0]
        self.voxel_size_y = voxel_size[1]

        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]

        self.min_radius = min_radius

    def gaussian_radius(
        self,
        height,
        width,
        min_overlap=0.7,
    ):
        """
        根据目标在 BEV 中的尺寸计算 Gaussian radius。
        """

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)

        sq1 = math.sqrt(
            b1 ** 2 - 4 * a1 * c1
        )

        r1 = (b1 - sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height

        sq2 = math.sqrt(
            b2 ** 2 - 4 * a2 * c2
        )

        r2 = (b2 - sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (
            (min_overlap - 1)
            * width
            * height
        )

        sq3 = math.sqrt(
            b3 ** 2 - 4 * a3 * c3
        )

        r3 = (b3 + sq3) / (2 * a3)

        return min(r1, r2, r3)

    def draw_gaussian(
        self,
        heatmap,
        center,
        radius,
    ):
        """
        在 heatmap 上画 Gaussian。

        Args:
            heatmap:
                [H, W]

            center:
                [x, y]

            radius:
                int
        """

        x, y = center

        diameter = 2 * radius + 1

        sigma = diameter / 6.0

        # -----------------------------------------
        # 创建 Gaussian
        # -----------------------------------------

        gaussian = torch.arange(
            diameter,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )

        gaussian = gaussian - radius

        yy, xx = torch.meshgrid(
            gaussian,
            gaussian,
            indexing="ij",
        )

        gaussian = torch.exp(
            -(xx ** 2 + yy ** 2)
            / (2 * sigma ** 2)
        )

        H, W = heatmap.shape

        # -----------------------------------------
        # Heatmap 范围
        # -----------------------------------------

        left = min(x, radius)
        right = min(
            W - x,
            radius + 1,
        )

        top = min(y, radius)
        bottom = min(
            H - y,
            radius + 1,
        )

        # -----------------------------------------
        # 如果中心有效
        # -----------------------------------------

        if (
            left <= 0
            or right <= 0
            or top <= 0
            or bottom <= 0
        ):
            return

        heatmap_slice = heatmap[
            y - top:y + bottom,
            x - left:x + right,
        ]

        gaussian_slice = gaussian[
            radius - top:radius + bottom,
            radius - left:radius + right,
        ]

        heatmap_slice[...] = torch.maximum(
            heatmap_slice,
            gaussian_slice,
        )

    def forward(
        self,
        gt_boxes,
        gt_labels,
        batch_size,
        num_classes,
        bev_h,
        bev_w,
    ):
        """
        Args:
            gt_boxes:
                [B, N, 9]

                [x, y, z, w, l, h, yaw, vx, vy]

            gt_labels:
                [B, N]

            batch_size:
                B

            num_classes:
                C

            bev_h:
                H

            bev_w:
                W

        Returns:
            targets:
                dict
        """

        device = gt_boxes.device
        dtype = gt_boxes.dtype

        # =========================================
        # 1. Heatmap
        # =========================================

        heatmap = torch.zeros(
            batch_size,
            num_classes,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        # =========================================
        # 2. Regression targets
        # =========================================

        offset = torch.zeros(
            batch_size,
            2,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        height = torch.zeros(
            batch_size,
            1,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        size = torch.zeros(
            batch_size,
            3,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        rotation = torch.zeros(
            batch_size,
            2,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        velocity = torch.zeros(
            batch_size,
            2,
            bev_h,
            bev_w,
            device=device,
            dtype=dtype,
        )

        # =========================================
        # 3. Regression mask
        # =========================================

        reg_mask = torch.zeros(
            batch_size,
            bev_h,
            bev_w,
            device=device,
            dtype=torch.bool,
        )

        # =========================================
        # 4. 遍历 batch
        # =========================================

        for b in range(batch_size):

            num_objects = gt_boxes.shape[1]

            for n in range(num_objects):

                box = gt_boxes[b, n]

                label = int(
                    gt_labels[b, n].item()
                )

                # ---------------------------------
                # 取 GT
                # ---------------------------------

                x = box[0]
                y = box[1]
                z = box[2]

                w = box[3]
                l = box[4]
                h = box[5]

                yaw = box[6]

                vx = box[7]
                vy = box[8]

                # ---------------------------------
                # 世界坐标 → BEV
                # ---------------------------------

                cx = (
                    x - self.x_min
                ) / self.voxel_size_x

                cy = (
                    y - self.y_min
                ) / self.voxel_size_y

                cx_int = int(cx)
                cy_int = int(cy)

                # ---------------------------------
                # 越界
                # ---------------------------------

                if (
                    cx_int < 0
                    or cx_int >= bev_w
                    or cy_int < 0
                    or cy_int >= bev_h
                ):
                    continue

                # ---------------------------------
                # BEV 中目标尺寸
                # ---------------------------------

                box_w = w / self.voxel_size_x
                box_l = l / self.voxel_size_y

                # ---------------------------------
                # Gaussian radius
                # ---------------------------------

                radius = self.gaussian_radius(
                    box_l,
                    box_w,
                )

                radius = max(
                    self.min_radius,
                    int(radius),
                )

                # ---------------------------------
                # Heatmap
                # ---------------------------------

                self.draw_gaussian(
                    heatmap[b, label],
                    (cx_int, cy_int),
                    radius,
                )

                # ---------------------------------
                # offset
                # ---------------------------------

                offset[
                    b,
                    0,
                    cy_int,
                    cx_int,
                ] = cx - cx_int

                offset[
                    b,
                    1,
                    cy_int,
                    cx_int,
                ] = cy - cy_int

                # ---------------------------------
                # height
                # ---------------------------------

                height[
                    b,
                    0,
                    cy_int,
                    cx_int,
                ] = z

                # ---------------------------------
                # size
                # ---------------------------------

                size[
                    b,
                    0,
                    cy_int,
                    cx_int,
                ] = w

                size[
                    b,
                    1,
                    cy_int,
                    cx_int,
                ] = l

                size[
                    b,
                    2,
                    cy_int,
                    cx_int,
                ] = h

                # ---------------------------------
                # rotation
                # ---------------------------------

                rotation[
                    b,
                    0,
                    cy_int,
                    cx_int,
                ] = torch.sin(yaw)

                rotation[
                    b,
                    1,
                    cy_int,
                    cx_int,
                ] = torch.cos(yaw)

                # ---------------------------------
                # velocity
                # ---------------------------------

                velocity[
                    b,
                    0,
                    cy_int,
                    cx_int,
                ] = vx

                velocity[
                    b,
                    1,
                    cy_int,
                    cx_int,
                ] = vy

                # ---------------------------------
                # mask
                # ---------------------------------

                reg_mask[
                    b,
                    cy_int,
                    cx_int,
                ] = True

        return {
            "heatmap": heatmap,
            "offset": offset,
            "height": height,
            "size": size,
            "rotation": rotation,
            "velocity": velocity,
            "reg_mask": reg_mask,
        }