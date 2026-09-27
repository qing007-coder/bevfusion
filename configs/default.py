class Config:
    """
    模型超参数配置。

    说明：
    - point_cloud_range 的顺序为 [x_min, y_min, z_min, x_max, y_max, z_max]。
    - 点云坐标系通常为：x 前向、y 左向、z 上向。
    - 长度单位默认为米（m）。
    """

    point_cloud_range = [
        0.0,   # x_min：点云 x 方向最小值，单位 m
        -20.0, # y_min：点云 y 方向最小值，单位 m
        -3.0,  # z_min：点云 z 方向最小值，单位 m
        40.0,  # x_max：点云 x 方向最大值，单位 m
        20.0,  # y_max：点云 y 方向最大值，单位 m
        1.0    # z_max：点云 z 方向最大值，单位 m
    ]

    voxel_size = [
        0.5,  # x 方向体素/柱体分辨率，单位 m
        0.5,  # y 方向体素/柱体分辨率，单位 m
        4.0   # z 方向体素/柱体高度，单位 m；可覆盖 z_min=-3 到 z_max=1
    ]

    max_points_per_pillar = 32  # 每个 pillar 最多保留的点数，超出则截断或采样
    max_pillars = 12000         # 单帧最多保留的 pillar 数量，用于限制计算量

    pillar_feature_dim = 64     # 每个 pillar 编码后输出的特征维度

    bev_h = 80                  # BEV 特征图高度，对应 y 方向栅格数：(20 - (-20)) / 0.5 = 80
    bev_w = 80                  # BEV 特征图宽度，对应 x 方向栅格数：(40 - 0) / 0.5 = 80

    bev_channels = 64           # BEV 特征图的通道数

    image_h = 256               # 输入相机图像高度，单位像素
    image_w = 512               # 输入相机图像宽度，单位像素

    camera_channels = 64        # 相机图像分支输出的特征通道数

    fusion_channels = 128       # 点云 BEV 特征与相机特征融合后的通道数

    num_classes = 3             # 检测类别数量，例如 Car、Pedestrian、Cyclist
    box_dim = 7                 # 3D 检测框维度：x, y, z, w, l, h, yaw