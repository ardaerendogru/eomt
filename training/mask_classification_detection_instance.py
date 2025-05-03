# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes  # Added import
from torchmetrics.detection import MeanAveragePrecision # Added import

from training.mask_detection_loss import MaskDetectionLoss
from training.lightning_module import LightningModule


class MaskClassificationDetectionInstance(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        hd_coefficient: float = 1.0,
        hd_alpha: float = 2.0,
        hd_k: int = 10,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
        finetuning_type: str = "all",
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
            finetuning_type=finetuning_type,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances

        self.criterion = MaskDetectionLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            hd_coefficient=hd_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            hd_alpha=hd_alpha,
            hd_k=hd_k,
        )

        self.init_metrics_detection(self.network.num_blocks + 1)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                # Handle empty masks for predictions
                valid_boxes = []
                valid_labels = []
                valid_scores = []
                
                for mask_idx, mask in enumerate(masks):
                    if mask.sum() > 0:  # Check if mask has any True values
                        try:
                            box = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
                            valid_boxes.append(box)
                            valid_labels.append(labels[mask_idx])
                            valid_scores.append(scores[mask_idx])
                        except RuntimeError:
                            continue
                
                # Create tensors from lists, or empty tensors if no valid masks
                if valid_boxes:
                    pred_boxes = torch.stack(valid_boxes)
                    pred_labels = torch.stack(valid_labels)
                    pred_scores = torch.stack(valid_scores)
                else:
                    pred_boxes = torch.zeros((0, 4), device=self.device)
                    pred_labels = torch.zeros(0, dtype=torch.int64, device=self.device)
                    pred_scores = torch.zeros(0, device=self.device)
                
                preds.append(
                    dict(
                        boxes=pred_boxes,
                        labels=pred_labels,
                        scores=pred_scores,
                    )
                )

                # For targets, we can directly convert since COCO always has valid boxes
                target_mask = masks_to_boxes(targets[j]["masks"])
                targets_.append(
                    dict(
                        boxes=target_mask,
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_detection(preds, targets_, i)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")