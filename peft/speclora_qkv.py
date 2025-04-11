import torch
import torch.nn as nn

class SpecLoRA_QKV(nn.Module):
    """
    Spectral LoRA implementation for QKV projection matrices in transformer attention layers.
    
    This applies the SpecLoRA approach specifically to Query and Value matrices,
    leaving the Key matrix unchanged, similar to the PiSSA implementation.
    """

    def __init__(
        self,
        qkv: nn.Module,
        rank: int,
        svd_k: int = None,  # Optional: use only top-k singular values/vectors
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        
        # Extract Q, K and V weights from the QKV projection
        with torch.no_grad():
            qkv_weight = self.qkv.weight
            q_weight = qkv_weight[:self.dim, :]  # [dim, dim]
            v_weight = qkv_weight[-self.dim:, :]  # [dim, dim]
            
            # Compute SVD for Q
            U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
            # Compute SVD for V
            U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)
            
            # Determine k (number of singular vectors to use)
            if svd_k is not None and svd_k < self.dim:
                self.k = svd_k
                self.U_q = U_q[:, :svd_k]
                self.V_q = Vh_q[:svd_k, :].T
                self.U_v = U_v[:, :svd_k]
                self.V_v = Vh_v[:svd_k, :].T
            else:
                self.k = self.dim
                self.U_q = U_q
                self.V_q = Vh_q.T
                self.U_v = U_v
                self.V_v = Vh_v.T
        
        # Initialize trainable parameters A and B for Q
        self.A_q = nn.Parameter(torch.zeros(self.k, rank))
        self.B_q = nn.Parameter(torch.zeros(rank, self.k))
        
        # Initialize trainable parameters A and B for V
        self.A_v = nn.Parameter(torch.zeros(self.k, rank))
        self.B_v = nn.Parameter(torch.zeros(rank, self.k))
        
        # Initialize A with small random values
        nn.init.normal_(self.A_q, std=0.01)
        nn.init.normal_(self.A_v, std=0.01)
        
    def forward(self, x) -> torch.Tensor:
        # Original QKV projection
        qkv_orig = self.qkv(x)
        
        # Split into query, key, value
        batch_size, seq_len = x.shape[0], x.shape[1]
        q_orig = qkv_orig[:, :, :self.dim]
        k_orig = qkv_orig[:, :, self.dim:2*self.dim]
        v_orig = qkv_orig[:, :, -self.dim:]
        
        # Reshape for matrix multiplication
        x_reshaped = x.view(-1, x.size(-1))  # [batch*seq, dim]
        
        # Compute spectral update for Q
        x_V_q = x_reshaped @ self.V_q  # Project input onto Q's right singular vectors
        x_V_B_q = x_V_q @ self.B_q.T  # Transform to rank-r space
        x_V_B_A_q = x_V_B_q @ self.A_q.T  # Transform to output singular vector space
        q_update = x_V_B_A_q @ self.U_q.T  # Project back to standard output space
        q_update = q_update.view(batch_size, seq_len, -1)  # Reshape back
        
        # Compute spectral update for V
        x_V_v = x_reshaped @ self.V_v  # Project input onto V's right singular vectors
        x_V_B_v = x_V_v @ self.B_v.T  # Transform to rank-r space
        x_V_B_A_v = x_V_B_v @ self.A_v.T  # Transform to output singular vector space
        v_update = x_V_B_A_v @ self.U_v.T  # Project back to standard output space
        v_update = v_update.view(batch_size, seq_len, -1)  # Reshape back
        
        # Add updates to original projections
        q_new = q_orig + q_update
        v_new = v_orig + v_update
        
        # Concatenate to form the new QKV
        return torch.cat((q_new, k_orig, v_new), dim=-1) 