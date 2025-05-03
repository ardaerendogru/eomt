import torch
import torch.nn as nn

class PiSSA(nn.Module):
    """PiSSA for the Query (Q) and Value (V) matrices of a combined QKV layer."""

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        
        # Ensure qkv is a Linear layer
        if not isinstance(qkv, nn.Linear):
            raise TypeError("qkv must be an instance of nn.Linear")
            
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features # Total output features (3 * dim)
        
        # Assuming q, k, v have the same dimension (common in ViTs)
        # and are stacked in the output dimension of the weight matrix
        if self.out_features % 3 != 0:
             raise ValueError("Output features must be divisible by 3 for QKV splitting.")
        self.dim = self.out_features // 3 
        
        if self.dim != self.in_features:
             # Warning or error depending on expected architecture
             print(f"Warning: Input features ({self.in_features}) and single head dim ({self.dim}) differ.")
             # Adjust self.dim if necessary based on architecture, but typically they match for qkv proj input/output per head dim

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_r

        # --- PiSSA Initialization ---
        qkv_weight = self.qkv.weight.data # Use .data to avoid grad history issues if any
        dtype = qkv_weight.dtype
        device = qkv_weight.device

        # Extract Q and V weights
        # Weight shape is typically (out_features, in_features) = (3 * dim, in_features)
        q_weight = qkv_weight[:self.dim, :]
        # k_weight = qkv_weight[self.dim : 2 * self.dim, :] # Not used by PiSSA here
        v_weight = qkv_weight[2 * self.dim :, :] # More robust slicing using 2*dim

        # Compute SVD for Q and V
        U_q, S_q, Vh_q = torch.linalg.svd(q_weight.float(), full_matrices=False) # SVD often more stable in float32
        U_v, S_v, Vh_v = torch.linalg.svd(v_weight.float(), full_matrices=False)

        # Move back to original dtype if needed after SVD
        U_q, S_q, Vh_q = U_q.to(dtype), S_q.to(dtype), Vh_q.to(dtype)
        U_v, S_v, Vh_v = U_v.to(dtype), S_v.to(dtype), Vh_v.to(dtype)

        # Create low-rank approximations using top-r singular values/vectors
        U_q_r = U_q[:, :self.lora_r]
        S_q_r = S_q[:self.lora_r]
        Vh_q_r = Vh_q[:self.lora_r, :]
        q_low_rank = U_q_r @ torch.diag(S_q_r) @ Vh_q_r

        U_v_r = U_v[:, :self.lora_r]
        S_v_r = S_v[:self.lora_r]
        Vh_v_r = Vh_v[:self.lora_r, :]
        v_low_rank = U_v_r @ torch.diag(S_v_r) @ Vh_v_r

        # Subtract low-rank approximations from original weights IN PLACE
        with torch.no_grad():
            self.qkv.weight[:self.dim, :] -= q_low_rank
            self.qkv.weight[2 * self.dim :, :] -= v_low_rank # Use same slicing as extraction

        # Initialize LoRA linear layers A and B for Q and V
        self.linear_a_q = nn.Linear(self.in_features, self.lora_r, bias=False, dtype=dtype, device=device)
        self.linear_b_q = nn.Linear(self.lora_r, self.dim, bias=False, dtype=dtype, device=device)
        self.linear_a_v = nn.Linear(self.in_features, self.lora_r, bias=False, dtype=dtype, device=device)
        self.linear_b_v = nn.Linear(self.lora_r, self.dim, bias=False, dtype=dtype, device=device)

        # Initialize weights of LoRA layers based on SVD components
        with torch.no_grad():
            # B = U * sqrt(S) -- weight shape (out, in) = (dim, r)
            self.linear_b_q.weight.copy_(U_q_r @ torch.diag(torch.sqrt(S_q_r)))
            # A = sqrt(S) * Vh -- weight shape (out, in) = (r, in_features)
            self.linear_a_q.weight.copy_(torch.diag(torch.sqrt(S_q_r)) @ Vh_q_r)

            # B = U * sqrt(S) -- weight shape (out, in) = (dim, r)
            self.linear_b_v.weight.copy_(U_v_r @ torch.diag(torch.sqrt(S_v_r)))
            # A = sqrt(S) * Vh -- weight shape (out, in) = (r, in_features)
            self.linear_a_v.weight.copy_(torch.diag(torch.sqrt(S_v_r)) @ Vh_v_r)

        # --- Freeze the original QKV layer ---
        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False
            
        # Ensure requires_grad is True for the new layers (should be by default)
        # for p in self.parameters(): # Check which params are trainable
        #     if p.requires_grad:
        #         print(f"Trainable param: shape {p.shape}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, in_features)

        # Residual QKV computation (using modified frozen weights)
        # Output shape: (batch_size, sequence_length, 3 * dim)
        qkv_residual = self.qkv(x)

        # Split the residual output
        q_res = qkv_residual[..., :self.dim]
        k_res = qkv_residual[..., self.dim : 2 * self.dim]
        v_res = qkv_residual[..., 2 * self.dim :]

        # Compute LoRA adjustments for Q and V
        # Input x shape: (B, N, in_features)
        # linear_a output: (B, N, r)
        # linear_b output: (B, N, dim)
        lora_q = self.linear_b_q(self.linear_a_q(x)) * self.scaling
        lora_v = self.linear_b_v(self.linear_a_v(x)) * self.scaling

        # Add adjustments
        q_new = q_res + lora_q
        # K remains unchanged (uses only the residual part)
        k_new = k_res
        v_new = v_res + lora_v

        # Concatenate back to QKV format
        # Output shape: (batch_size, sequence_length, 3 * dim)
        return torch.cat((q_new, k_new, v_new), dim=-1)

    # Optional: method to easily get trainable parameters
    def get_trainable_parameters(self):
        return list(self.linear_a_q.parameters()) + \
               list(self.linear_b_q.parameters()) + \
               list(self.linear_a_v.parameters()) + \
               list(self.linear_b_v.parameters())