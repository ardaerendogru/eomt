import torch
import torch.nn as nn
import math

class SpecLoRA(nn.Module):
    """
    Spectral LoRA implementation for QKV projection matrices in transformer attention layers.
    
    This applies the SpecLoRA approach specifically to Query and Value matrices,
    leaving the Key matrix unchanged, similar to the PiSSA implementation.
    """

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int,
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = lora_r
        
        
        # Extract Q, K and V weights from the QKV projection
        with torch.no_grad():
            qkv_weight = self.qkv.weight
            q_weight = qkv_weight[:self.dim, :]  # [dim, dim]
            v_weight = qkv_weight[-self.dim:, :]  # [dim, dim]
            
            # Compute SVD for Q
            U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
            # Compute SVD for V
            U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)
            
            self.k = self.dim
            # Store SVD results as nn.Parameters to ensure they're on the correct device
            self.register_buffer('U_q', U_q)  # [dim, dim]
            self.register_buffer('V_q', Vh_q.T)  # [dim, dim]
            self.register_buffer('U_v', U_v)  # [dim, dim]
            self.register_buffer('V_v', Vh_v.T)  # [dim, dim]
    
        # Initialize LoRA-style linear layers for Q
        self.linear_a_q = nn.Linear(self.dim, lora_r, bias=False)  # Input mapped to V_q basis, then to rank-r space
        self.linear_b_q = nn.Linear(lora_r, self.dim, bias=False)  # Rank-r space mapped to U_q basis
        
        # Initialize LoRA-style linear layers for V
        self.linear_a_v = nn.Linear(self.dim, lora_r, bias=False)  # Input mapped to V_v basis, then to rank-r space
        self.linear_b_v = nn.Linear(lora_r, self.dim, bias=False)  # Rank-r space mapped to U_v basis
        
        # Initialize A layers with Kaiming uniform
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        
        # Initialize B layers to zero
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
        
    def forward(self, x) -> torch.Tensor:
        # Original QKV projection
        qkv_orig = self.qkv(x)
        
        # Split into query, key, value
        batch_size, seq_len = x.shape[0], x.shape[1]
        q_orig = qkv_orig[:, :, :self.dim]
        k_orig = qkv_orig[:, :, self.dim:2*self.dim]
        v_orig = qkv_orig[:, :, -self.dim:]
        
        x_reshaped = x.view(-1, x.size(-1))  # [batch*seq, dim]
        
        # Spectral projection for Q
        x_V_q = x_reshaped @ self.V_q  # Project input onto Q's right singular vectors
        q_r = self.linear_a_q(x_V_q)  # Transform to rank-r space
        q_update_spectral = self.linear_b_q(q_r)  # Transform to left singular vector space
        q_update = q_update_spectral @ self.U_q.T  # Project back to standard output space
        q_update = q_update.view(batch_size, seq_len, -1)
        
        # Spectral projection for V
        x_V_v = x_reshaped @ self.V_v  # Project input onto V's right singular vectors
        v_r = self.linear_a_v(x_V_v)  # Transform to rank-r space
        v_update_spectral = self.linear_b_v(v_r)  # Transform to left singular vector space
        v_update = v_update_spectral @ self.U_v.T  # Project back to standard output space
        v_update = v_update.view(batch_size, seq_len, -1) 
        
        # Add updates to original projections
        q_new = q_orig + q_update
        v_new = v_orig + v_update
        
        # Concatenate to form the new QKV
        return torch.cat((q_new, k_orig, v_new), dim=-1) 