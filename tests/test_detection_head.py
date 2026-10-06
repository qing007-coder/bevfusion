import os
import sys

import torch

# 允许直接运行: python tests/test_detection_head.py
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from models.heads.detection_head import CenterHead


def main():

    B = 2
    C = 256
    H = 200
    W = 200

    # BEV Fusion 输出
    fused_bev = torch.randn(
        B,
        C,
        H,
        W
    )

    model = CenterHead(
        in_channels=256,
        num_classes=10,
        hidden_channels=256,
    )

    outputs = model(fused_bev)

    for name, value in outputs.items():
        print(
            f"{name:10s}: {value.shape}"
        )


if __name__ == "__main__":
    main()