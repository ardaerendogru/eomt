import torch
import torch.nn as nn

class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int, # Add lora_r to initialize linear layers
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_r = lora_r # Store lora_r

        # Initialize LoRA linear layers
        self.linear_a_q = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_q = nn.Linear(lora_r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_v = nn.Linear(lora_r, self.dim, bias=False)

        # Initialize linear_b layers to zero
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)


        # self.w_identity = torch.eye(self.dim) # Not used in forward

    def forward(self, x) -> torch.Tensor:
        # Example alternative (less memory efficient, potentially more compiler friendly)
        qkv_orig = self.qkv(x)
        q_orig = qkv_orig[:, :, :self.dim]
        k_orig = qkv_orig[:, :, self.dim:2*self.dim] # Assuming K is in the middle
        v_orig = qkv_orig[:, :, -self.dim:]

        lora_q = self.linear_b_q(self.linear_a_q(x))
        lora_v = self.linear_b_v(self.linear_a_v(x))

        q_new = q_orig + lora_q
        v_new = v_orig + lora_v

        return torch.cat((q_new, k_orig, v_new), dim=-1)