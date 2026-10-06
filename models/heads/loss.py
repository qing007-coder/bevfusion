import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPointLoss(nn.Module):

    def __init__(
        self,
        heatmap_weight=1.0,
        offset_weight=1.0,
        height_weight=1.0,
        size_weight=1.0,
        rotation_weight=1.0,
        velocity_weight=1.0,
    ):
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.height_weight = height_weight
        self.size_weight = size_weight
        self.rotation_weight = rotation_weight
        self.velocity_weight = velocity_weight

    # ============================================================
    # 1. Heatmap Loss
    # ============================================================

    def heatmap_loss(self, pred, target):

        """
        pred:
            [B, C, H, W]

        target:
            [B, C, H, W]

        使用 CenterNet / CenterPoint 风格的 Gaussian focal loss
        """

        pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        neg_weights = torch.pow(
            1 - target,
            4
        )

        pos_loss = (
            torch.log(pred)
            * torch.pow(1 - pred, 2)
            * pos_mask
        )

        neg_loss = (
            torch.log(1 - pred)
            * torch.pow(pred, 2)
            * neg_weights
            * neg_mask
        )

        num_pos = pos_mask.sum()

        if num_pos > 0:

            loss = -(
                pos_loss.sum()
                + neg_loss.sum()
            ) / num_pos

        else:

            loss = -neg_loss.sum()

        return loss

    # ============================================================
    # 2. Masked L1
    # ============================================================

    def masked_l1_loss(
        self,
        pred,
        target,
        mask,
    ):

        """
        pred:
            [B, C, H, W]

        target:
            [B, C, H, W]

        mask:
            [B, H, W]
        """

        mask = mask.unsqueeze(1).float()

        loss = F.l1_loss(
            pred * mask,
            target * mask,
            reduction="sum",
        )

        normalizer = mask.sum() * pred.shape[1]

        if normalizer > 0:
            loss = loss / normalizer

        return loss

    # ============================================================
    # 3. Forward
    # ============================================================

    def forward(
        self,
        predictions,
        targets,
    ):

        pred_heatmap = torch.sigmoid(
            predictions["heatmap"]
        )

        pred_offset = predictions["offset"]
        pred_height = predictions["height"]
        pred_size = predictions["size"]
        pred_rotation = predictions["rotation"]
        pred_velocity = predictions["velocity"]

        target_heatmap = targets["heatmap"]
        target_offset = targets["offset"]
        target_height = targets["height"]
        target_size = targets["size"]
        target_rotation = targets["rotation"]
        target_velocity = targets["velocity"]

        reg_mask = targets["reg_mask"]

        # --------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------

        loss_heatmap = self.heatmap_loss(
            pred_heatmap,
            target_heatmap,
        )

        # --------------------------------------------------------
        # Regression
        # --------------------------------------------------------

        loss_offset = self.masked_l1_loss(
            pred_offset,
            target_offset,
            reg_mask,
        )

        loss_height = self.masked_l1_loss(
            pred_height,
            target_height,
            reg_mask,
        )

        loss_size = self.masked_l1_loss(
            pred_size,
            target_size,
            reg_mask,
        )

        loss_rotation = self.masked_l1_loss(
            pred_rotation,
            target_rotation,
            reg_mask,
        )

        loss_velocity = self.masked_l1_loss(
            pred_velocity,
            target_velocity,
            reg_mask,
        )

        # --------------------------------------------------------
        # Total
        # --------------------------------------------------------

        total_loss = (
            self.heatmap_weight * loss_heatmap
            + self.offset_weight * loss_offset
            + self.height_weight * loss_height
            + self.size_weight * loss_size
            + self.rotation_weight * loss_rotation
            + self.velocity_weight * loss_velocity
        )

        return {
            "loss": total_loss,
            "loss_heatmap": loss_heatmap,
            "loss_offset": loss_offset,
            "loss_height": loss_height,
            "loss_size": loss_size,
            "loss_rotation": loss_rotation,
            "loss_velocity": loss_velocity,
        }