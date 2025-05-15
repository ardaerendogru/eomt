# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


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
        lora_class=None,
        lora_r=None,
        lora_alpha=None,
        num_lora_free_blocks=None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )
        self.lora_class = lora_class
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_free_blocks = num_lora_free_blocks
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

    def apply_lora(self):

        if self.lora_class is not None:
            # Freeze all parameters initially
            for param in self.backbone.parameters():
                param.requires_grad = False

            total_blocks = len(self.backbone.blocks)
            num_lora_blocks = total_blocks - self.lora_free_blocks
            if num_lora_blocks < 0:
                raise ValueError(f"num_lora_free_blocks ({self.lora_free_blocks}) cannot be greater than total blocks ({total_blocks})")

            # Select LoRA class based on configuration
            lora_implementation = None
            if self.lora_class == "LoRA":
                lora_implementation = LoRA
            elif self.lora_class == "PiSSA":
                lora_implementation = PiSSA
            elif self.lora_class == "SpecLoRA":
                lora_implementation = SpecLoRA
            elif self.lora_class == "ProLoRA":
                lora_implementation = ProLoRA
            else:
                raise ValueError(f"Unsupported LoRA class: {self.lora_class}. Use 'LoRA' or 'PiSSA'.")

            # Apply configured LoRA to specified blocks
            for i in range(num_lora_blocks):
                block = self.backbone.blocks[i]
                qkv_layer = block.attn.qkv  # Assuming standard timm naming
                lora_layer = lora_implementation(qkv_layer, lora_r=self.lora_r, lora_alpha=self.lora_alpha)
                block.attn.qkv = lora_layer  # Replace the original qkv layer
                
            # Unfreeze parameters in the final blocks (non-LoRA blocks)
            for i in range(num_lora_blocks, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

            print(f"Applied {self.lora_class} to the first {num_lora_blocks} blocks with rank {self.lora_r}.")
            # Add a method to print trainable parameters if needed, similar to peft's
            self.print_trainable_parameters()
