# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import timm
import torch
import torch.nn as nn
from peft import LoRA

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        use_lora: bool = False,
        lora_r: int = 32,
        num_lora_free_blocks: int = 4,  # Number of final blocks to exclude from LoRA
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )

        self.use_lora = use_lora
        if self.use_lora:
             # Freeze all parameters initially
            for param in self.backbone.parameters():
                param.requires_grad = False

            total_blocks = len(self.backbone.blocks)
            num_lora_blocks = total_blocks - num_lora_free_blocks
            if num_lora_blocks < 0:
                raise ValueError(f"num_lora_free_blocks ({num_lora_free_blocks}) cannot be greater than total blocks ({total_blocks})")

            # Apply custom LoRA to specified blocks
            for i in range(num_lora_blocks):
                block = self.backbone.blocks[i]
                qkv_layer = block.attn.qkv # Assuming standard timm naming
                lora_layer = LoRA(qkv_layer, lora_r=lora_r)
                block.attn.qkv = lora_layer # Replace the original qkv layer
            # Unfreeze parameters in the final blocks (non-LoRA blocks)
            for i in range(num_lora_blocks, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

            print(f"Applied custom LoRA to the first {num_lora_blocks} blocks.")
            # Add a method to print trainable parameters if needed, similar to peft's
            self.print_trainable_parameters()


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

