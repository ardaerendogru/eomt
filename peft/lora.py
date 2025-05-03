import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_r = lora_r
        self.scaling = lora_alpha / lora_r
        
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # Initialize LoRA linear layers
        self.linear_a_q = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_q = nn.Linear(lora_r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_v = nn.Linear(    lora_r, self.dim, bias=False)

        # Initialize linear_a layers with Kaiming uniform
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        
        # Initialize linear_b layers to zero
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x) -> torch.Tensor:
        # Get original QKV output
        qkv_orig = self.qkv(x)
        q_orig = qkv_orig[:, :, :self.dim]
        k_orig = qkv_orig[:, :, self.dim:2*self.dim]
        v_orig = qkv_orig[:, :, -self.dim:]
        
        # Apply dropout before LoRA computation
        x_dropped = self.lora_dropout(x)
        
        # Compute LoRA contributions with scaling
        lora_q = self.linear_b_q(self.linear_a_q(x_dropped)) * self.scaling
        lora_v = self.linear_b_v(self.linear_a_v(x_dropped)) * self.scaling
        
        # Add LoRA contributions to original outputs
        q_new = q_orig + lora_q
        v_new = v_orig + lora_v
        
        # Concatenate and return the result
        return torch.cat((q_new, k_orig, v_new), dim=-1)