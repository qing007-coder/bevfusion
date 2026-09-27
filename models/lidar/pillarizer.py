import torch


class Pillarizer:
    """
    将 Batch 点云划分为 Pillar。

    输入:
        points:
            [B, N, 4]
            B = batch size
            N = 每个样本的点数
            4 = [x, y, z, intensity]

    输出:
        pillars:
            [N_total, max_points_per_pillar, 4]
            N_total = 整个 Batch 中实际保留下来的非空 Pillar 数量

        coords:
            [N_total, 3]
            每个 Pillar 的坐标:
            [batch_id, pillar_y, pillar_x]
    """

    def __init__(
            self,
            point_cloud_range,
            voxel_size,
            max_points_per_pillar=32,
            max_pillars=12000,
    ):
        """
        Args:
            point_cloud_range:
                点云空间范围。

                格式:
                [x_min, y_min, z_min,
                 x_max, y_max, z_max]

            voxel_size:
                Pillar 在 XY 平面上的尺寸。

                格式:
                [voxel_x, voxel_y, voxel_z]

                本实现只使用:
                voxel_x
                voxel_y

            max_points_per_pillar:
                每个 Pillar 最多保存多少个点。

            max_pillars:
                每个 Batch 样本最多保留多少个非空 Pillar。

        Returns:
            None
        """

        # Point Cloud Range
        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]

        self.x_max = point_cloud_range[3]
        self.y_max = point_cloud_range[4]

        # Voxel / Pillar Size
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]


        # Pillar 限制
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # BEV Spatial Shape
        # X 方向有多少个 Pillar
        self.bev_w = int(
            (self.x_max - self.x_min) / self.vx
        )

        # Y 方向有多少个 Pillar
        self.bev_h = int(
            (self.y_max - self.y_min) / self.vy
        )

        # 后面 Scatter 会使用
        self.spatial_shape = (
            self.bev_h,
            self.bev_w
        )

    def __call__(self, points):
        """
        Args:
            points:
                [B, N, 4]

                B = Batch Size
                N = 每个样本的点数
                4 = [x, y, z, intensity]

        Returns:
            pillars:
                [N_total, max_points_per_pillar, 4]

                N_total = 整个 Batch 中实际保留的
                          非空 Pillar 总数量。

            coords:
                [N_total, 3]

                每个 Pillar 的坐标:
                [batch_id, pillar_y, pillar_x]
        """

        B, _, _ = points.shape

        all_pillars = []
        all_coords = []

        # 分别处理 Batch 中的每一个样本
        for batch_id in range(B):

            pillars, coords = self._handle_single_batch(
                points[batch_id]
            )

            # 给当前样本的 Pillar 坐标增加 batch_id
            batch_column = torch.full(
                (coords.shape[0], 1),
                batch_id,
                dtype=torch.long,
                device=points.device
            ) # [M, 1] 

            coords = torch.cat(
                [
                    batch_column,
                    coords
                ],
                dim=1
            ) # [M, 2] 和 [M, 1] 拼接 [M, 3]

            all_pillars.append(pillars)
            all_coords.append(coords)

        # 合并整个 Batch

        if len(all_pillars) == 0:

            pillars = torch.zeros(
                0,
                self.max_points_per_pillar,
                4,
                dtype=points.dtype,
                device=points.device
            )

            coords = torch.zeros(
                0,
                3,
                dtype=torch.long,
                device=points.device
            )

            return pillars, coords

        # 这么拼接 每个pillar_nums是不一样的 但是pillar有coords来当作索引 查找所在batch, x, y
        pillars = torch.cat(
            all_pillars,
            dim=0
        )

        coords = torch.cat(
            all_coords,
            dim=0
        )

        return pillars, coords

    def _handle_single_batch(self, points):
        """
        处理单个 Batch 样本的点云，
        将 Point Cloud 划分成 Pillar。

        Args:
            points:
                [N, 4]

                N = 当前样本的点数量
                4 = [x, y, z, intensity]

        Returns:
            pillar_points:
                [M, max_points_per_pillar, 4]

                M = 当前样本实际保留下来的
                    非空 Pillar 数量。

            pillar_coords:
                [M, 2]

                每个 Pillar 的坐标:

                [pillar_y, pillar_x]
        """

        _, C = points.shape

        # 过滤空间范围之外的点
        mask = (
            (points[:, 0] >= self.x_min) &
            (points[:, 1] >= self.y_min) &
            (points[:, 0] < self.x_max) &
            (points[:, 1] < self.y_max)
        )

        points = points[mask]

        # 当前样本没有有效点
        if points.shape[0] == 0:

            return (
                torch.zeros(
                    0,
                    self.max_points_per_pillar,
                    C,
                    dtype=points.dtype,
                    device=points.device
                ),

                torch.zeros(
                    0,
                    2,
                    dtype=torch.long,
                    device=points.device
                )
            )


        # 计算每个点所在Pillar
        # X 方向坐标
        pillar_x = (
            (points[:, 0] - self.x_min) / self.vx
        ).long()

        # Y 方向坐标
        pillar_y = (
            (points[:, 1] - self.y_min) / self.vy
        ).long() # [N]

        pillar_coords = torch.stack(
            [
                pillar_y,
                pillar_x
            ],
            dim=1
        ) # [N, 2]

        # 找到所有非空 Pillar
        unique_coords, inverse = torch.unique(
            pillar_coords,
            dim=0,
            return_inverse=True
        ) # unique_coords:[M, 2] inverse:[N] inverse负责对应N和M的关系


        # 限制最大 Pillar 数量
        pillar_nums = min(
            self.max_pillars,
            unique_coords.shape[0]
        )
        unique_coords = unique_coords[:pillar_nums]

  
        # 创建 Pillar Tensor
        pillar_points = torch.zeros(
            pillar_nums,
            self.max_points_per_pillar,
            C,
            dtype=points.dtype,
            device=points.device
        )

        # 记录每个 Pillar 当前已经放了多少个点

        pillar_count = torch.zeros(
            pillar_nums,
            dtype=torch.long,
            device=points.device
        )

        # 将 Point 放入对应 Pillar
        for i in range(points.shape[0]):

            # 当前点属于哪个 Pillar
            pillar_id = inverse[i].item()

            # 如果这个 Pillar 超过 max_pillars，
            # 直接丢弃其中的点。

            if pillar_id >= pillar_nums:
                continue

            # 当前 Pillar 已经有多少个点
            count = pillar_count[pillar_id].item()

            # 超过单个 Pillar 最大点数
            if count >= self.max_points_per_pillar:
                continue

            # 将当前点放入 Pillar
            pillar_points[
                pillar_id,
                count
            ] = points[i]

            pillar_count[pillar_id] += 1

        return pillar_points, unique_coords
