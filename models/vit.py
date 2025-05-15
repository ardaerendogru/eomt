# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import timm
import torch
import torch.nn as nn
# from peft import LoRA, PiSSA, SpecLoRA, ProLoRA

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    # Add a helper method to print trainable parameters
    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.backbone.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

