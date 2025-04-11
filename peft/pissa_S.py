import torch
import torch.nn as nn

class PiSSA_S(nn.Module):
    """PiSSA for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int, # Add lora_r to initialize linear layers
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_r = lora_r # Store lora_r

        # Extract Q, K and V weights from the QKV projection
        qkv_weight = self.qkv.weight
        q_weight = qkv_weight[:self.dim, :]
        # k_weight = qkv_weight[self.dim:2*self.dim, :]
        v_weight = qkv_weight[-self.dim:, :]
        
        # Compute SVD for Q and V
        U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
        U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)


        # Subtract low-rank approximations from original weights
        with torch.no_grad():
            self.qkv.weight[:self.dim, :] = q_weight - q_low_rank
            self.qkv.weight[-self.dim:, :] = v_weight - v_low_rank


        # Initialize LoRA linear layers
        self.linear_a_q = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_q = nn.Linear(lora_r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_v = nn.Linear(lora_r, self.dim, bias=False)

        with torch.no_grad():
            self.linear_a_q.weight.copy_(torch.diag(torch.sqrt(S_q_r)) @ Vh_q_r)
            self.linear_b_q.weight.copy_(U_q_r @ torch.diag(torch.sqrt(S_q_r)))
            
            self.linear_a_v.weight.copy_(torch.diag(torch.sqrt(S_v_r)) @ Vh_v_r)
            self.linear_b_v.weight.copy_(U_v_r @ torch.diag(torch.sqrt(S_v_r)))

    def forward(self, x) -> torch.Tensor:
        
        qkv_orig = self.qkv(x)
        q_orig = qkv_orig[:, :, :self.dim]
        k_orig = qkv_orig[:, :, self.dim:2*self.dim] 
        v_orig = qkv_orig[:, :, -self.dim:]

        lora_q = self.linear_b_q(self.linear_a_q(x))
        lora_v = self.linear_b_v(self.linear_a_v(x))

        q_new = q_orig + lora_q
        v_new = v_orig + lora_v

        return torch.cat((q_new, k_orig, v_new), dim=-1)
