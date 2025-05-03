# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# 
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.scale_block import ScaleBlock
from peft import LoRA, PiSSA, SpecLoRA, ProLoRA

class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        lora_class=None,
        lora_r=None,
        num_lora_free_blocks=None,
        lora_layers=None,
        lora_alpha=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.lora_class = lora_class
        self.lora_r = lora_r
        self.lora_free_blocks = num_lora_free_blocks
        self.lora_layers = lora_layers
        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )


    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.backbone.patch_embed(x)
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.patch_drop(x)
        x = self.encoder.backbone.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    self.encoder.backbone.patch_embed.grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : self.num_q,
                    self.num_q + self.encoder.backbone.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[
                        i - len(self.encoder.backbone.blocks) + self.num_blocks
                    ],
                )

            x = x + block.drop_path1(
                block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
            )
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
    
    
    def apply_lora(self):
        #TODO: Fix this, you need to seperate qkv and then apply lora to selected ones
        if self.lora_class is not None:
            # Freeze all parameters initially
            # Keep track of which top-level modules have LoRA applied
            lora_applied_to_modules = set()
            for param in self.parameters(): # Freeze everything first
                param.requires_grad = False


            total_blocks = len(self.encoder.backbone.blocks)
            # Determine the number of blocks to apply LoRA to
            num_lora_blocks = total_blocks
            if self.lora_free_blocks is not None:
                 num_lora_blocks = total_blocks - self.lora_free_blocks
                 if num_lora_blocks < 0:
                     raise ValueError(f"num_lora_free_blocks ({self.lora_free_blocks}) cannot be greater than total blocks ({total_blocks})")

            if not self.lora_layers:
                raise ValueError("lora_layers must be specified when lora_class is set.")

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
                raise ValueError(f"Unsupported LoRA class: {self.lora_class}. Supported: LoRA, PiSSA, SpecLoRA, ProLoRA.")

            print(f"Attempting to apply {self.lora_class} (r={self.lora_r}) to layers {self.lora_layers} in the first {num_lora_blocks} blocks.")

            # Apply configured LoRA to specified transformer blocks and layers
            for i in range(num_lora_blocks):
                block = self.encoder.backbone.blocks[i]
                applied_layers_in_block = []

                # Filter lora_layers that are relevant for transformer blocks
                block_specific_layers = [lname for lname in self.lora_layers if lname not in ['mask_head']] # Add other non-block keys if needed

                for layer_name in block_specific_layers:
                    target_layer = None
                    parent_module = None
                    attr_name = None
                    layer_applied_msg = layer_name # Default message part

                    try:
                        if layer_name == 'qkv':
                            target_layer = block.attn.qkv
                            parent_module = block.attn
                            attr_name = 'qkv'
                        elif layer_name == 'q':
                            # Try separate q, else handle combined qkv
                            if hasattr(block.attn, 'q') and isinstance(block.attn.q, nn.Linear):
                                target_layer = block.attn.q
                                parent_module = block.attn
                                attr_name = 'q'
                            elif hasattr(block.attn, 'qkv'):
                                print(f"Warning: Block {i} has combined 'qkv'. Applying {self.lora_class} to 'qkv' for target 'q'.")
                                target_layer = block.attn.qkv
                                parent_module = block.attn
                                attr_name = 'qkv'
                                layer_applied_msg = 'qkv (for q)'
                            else:
                                raise AttributeError("Neither 'q' nor 'qkv' found")
                        elif layer_name == 'k':
                             # Try separate k, else handle combined qkv
                            if hasattr(block.attn, 'k') and isinstance(block.attn.k, nn.Linear):
                                target_layer = block.attn.k
                                parent_module = block.attn
                                attr_name = 'k'
                            elif hasattr(block.attn, 'qkv'):
                                print(f"Warning: Block {i} has combined 'qkv'. Applying {self.lora_class} to 'qkv' for target 'k'.")
                                target_layer = block.attn.qkv
                                parent_module = block.attn
                                attr_name = 'qkv'
                                layer_applied_msg = 'qkv (for k)'
                            else:
                                raise AttributeError("Neither 'k' nor 'qkv' found")
                        elif layer_name == 'v':
                             # Try separate v, else handle combined qkv
                            if hasattr(block.attn, 'v') and isinstance(block.attn.v, nn.Linear):
                                target_layer = block.attn.v
                                parent_module = block.attn
                                attr_name = 'v'
                            elif hasattr(block.attn, 'qkv'):
                                print(f"Warning: Block {i} has combined 'qkv'. Applying {self.lora_class} to 'qkv' for target 'v'.")
                                target_layer = block.attn.qkv
                                parent_module = block.attn
                                attr_name = 'qkv'
                                layer_applied_msg = 'qkv (for v)'
                            else:
                                raise AttributeError("Neither 'v' nor 'qkv' found")
                        elif layer_name == 'proj':
                            target_layer = block.attn.proj
                            parent_module = block.attn
                            attr_name = 'proj'
                        elif layer_name == 'ffn1': # Assuming FFN/MLP uses fc1, fc2
                            target_layer = block.mlp.fc1
                            parent_module = block.mlp
                            attr_name = 'fc1'
                            layer_applied_msg = 'ffn1 (fc1)'
                        elif layer_name == 'ffn2':
                            target_layer = block.mlp.fc2
                            parent_module = block.mlp
                            attr_name = 'fc2'
                            layer_applied_msg = 'ffn2 (fc2)'
                        else:
                            print(f"Warning: Unsupported block LoRA layer name '{layer_name}' in block {i}. Skipping.")
                            continue

                        # Check if target_layer is actually a Linear layer suitable for LoRA
                        if not isinstance(target_layer, nn.Linear):
                            print(f"Warning: Target layer '{layer_name}' ({attr_name}) in block {i} is not nn.Linear. Skipping LoRA application.")
                            continue

                        # Check if LoRA already applied to this specific layer/attribute in this block
                        current_layer = getattr(parent_module, attr_name)
                        if self.lora_class in str(type(current_layer)):
                             print(f"Note: LoRA already applied to {attr_name} in block {i}. Skipping redundant application for target '{layer_name}'.")
                             continue

                        # Apply LoRA
                        lora_layer = lora_implementation(target_layer, lora_r=self.lora_r)
                        setattr(parent_module, attr_name, lora_layer) # Replace the original layer
                        applied_layers_in_block.append(layer_applied_msg)
                        lora_applied_to_modules.add('encoder.backbone') # Mark backbone as modified

                    except AttributeError as e:
                        print(f"Warning: Could not find layer corresponding to '{layer_name}' in block {i}: {e}. Skipping.")
                        continue

                if applied_layers_in_block:
                     unique_applied = sorted(list(set(applied_layers_in_block)))
                     print(f"Block {i}: Applied {self.lora_class} to: {', '.join(unique_applied)}")

            # Apply LoRA to mask_head if specified
            if 'mask_head' in self.lora_layers:
                print(f"Applying {self.lora_class} (r={self.lora_r}) to nn.Linear layers in mask_head.")
                applied_mask_head = False
                for idx, layer in enumerate(self.mask_head):
                    if isinstance(layer, nn.Linear):
                        # Check if already wrapped
                        if self.lora_class not in str(type(layer)):
                            lora_layer = lora_implementation(layer, lora_r=self.lora_r)
                            self.mask_head[idx] = lora_layer # Replace layer in Sequential
                            applied_mask_head = True
                        else:
                            print(f"Note: LoRA already applied to mask_head layer index {idx}. Skipping.")

                if applied_mask_head:
                    lora_applied_to_modules.add('mask_head')
                    print("Finished applying LoRA to mask_head.")
                else:
                     print("No new LoRA layers applied to mask_head (potentially already applied or no Linear layers).")


            # Unfreeze parameters in the final blocks (non-LoRA blocks)
            if self.lora_free_blocks is not None and self.lora_free_blocks > 0:
                for i in range(num_lora_blocks, total_blocks):
                    print(f"Unfreezing parameters in block {i} (non-LoRA block).")
                    for param in self.encoder.backbone.blocks[i].parameters():
                        param.requires_grad = True
                    lora_applied_to_modules.add('encoder.backbone') # Mark as modified if any part is trainable


            # Also unfreeze other specified components *if LoRA was not applied to them*
            print("Checking components to unfreeze (if LoRA not applied)...")
            modules_to_unfreeze = {
                'class_head': self.class_head,
                'mask_head': self.mask_head, # Will be skipped if LoRA applied
                'upscale': self.upscale,
                'q': self.q
            }

            for name, module in modules_to_unfreeze.items():
                 if name not in lora_applied_to_modules:
                     print(f"Unfreezing {name} parameters.")
                     for param in module.parameters():
                         param.requires_grad = True
                 else:
                     print(f"Skipping unfreeze for {name} as LoRA was applied to it (or its submodules).")


            # Always make LoRA parameters trainable (redundant if handled by PEFT wrapper, but safe)
            print("Ensuring all LoRA adapter parameters are trainable...")
            for name, param in self.named_parameters():
                if 'lora_' in name or '.lora.' in name: # Common LoRA parameter naming conventions
                    param.requires_grad = True

            print(f"\nFinished applying {self.lora_class} (r={self.lora_r}).")
            self.print_trainable_parameters()


    # Placeholder for the print_trainable_parameters method if you need it
    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters(): # Use named_parameters for clarity
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # Optional: Print names of trainable params for debugging
                # print(f"  Trainable: {name} ({param.numel()})")
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
