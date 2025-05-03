# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)
from .hausdorff_er_loss import HausdorffERLoss


class MaskDetectionLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        hd_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        hd_alpha: float = 2.0,
        hd_k: int = 10,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.hd_coefficient = hd_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

        self.hd_loss_func = HausdorffERLoss(alpha=hd_alpha, k=hd_k, reduction='mean')

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        total_hd_loss = 0.0
        num_pairs = 0
        device = masks_queries_logits.device

        for i, (pred_indices, target_indices) in enumerate(indices):
            if len(pred_indices) == 0:
                continue

            for pred_idx, target_idx in zip(pred_indices, target_indices):
                pred_mask_logit = masks_queries_logits[i, pred_idx]
                pred_mask_prob = torch.sigmoid(pred_mask_logit).unsqueeze(0).unsqueeze(0)

                target_mask = mask_labels[i][target_idx]
                target_mask_long = target_mask.long().unsqueeze(0).unsqueeze(0)

                pair_hd_loss = self.hd_loss_func(pred_mask_prob, target_mask_long)
                total_hd_loss += pair_hd_loss
                num_pairs += 1

        loss_hd = total_hd_loss / (num_pairs + 1e-8) if num_pairs > 0 else torch.tensor(0.0, device=device)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        num_masks_norm = torch.clamp(num_masks_tensor / world_size, min=1)
        loss_hd_normalized = loss_hd / num_masks_norm

        return {
            **loss_masks,
            **loss_classes,
            "loss_hd": loss_hd_normalized
        }

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_key = f"train_{loss_key}"
            log_fn(log_key, loss, sync_dist=True)

            weight = 0.0
            if "mask" in loss_key:
                weight = self.mask_coefficient
            elif "dice" in loss_key:
                weight = self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weight = self.class_coefficient
            elif "hd" in loss_key:
                weight = self.hd_coefficient
            else:
                raise ValueError(f"Unknown loss key pattern: {loss_key}")

            weighted_loss = loss * weight

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        if loss_total is None:
            example_tensor = next(iter(losses_all_layers.values()))
            loss_total = torch.tensor(0.0, device=example_tensor.device, dtype=example_tensor.dtype, requires_grad=True)

        log_fn("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
