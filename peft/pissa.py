import torch
import torch.nn as nn
import torch.nn.functional as F # Add functional import

class PiSSA(nn.Module):
    """PiSSA for the for Query (Q), Key (Q), Value (V) matrices
    with learnable singular values for the residual components."""

    def __init__(
        self,
        qkv: nn.Module,
        lora_r: int, # Add lora_r to initialize linear layers
    ):
        super().__init__()
        # self.qkv = qkv # We won't need the original layer directly after init
        self.dim = qkv.in_features
        self.lora_r = lora_r # Store lora_r

        # Extract Q, K and V weights from the QKV projection
        qkv_weight = qkv.weight.detach() # Use detach()
        q_weight = qkv_weight[:self.dim, :]
        k_weight = qkv_weight[self.dim:2*self.dim, :]
        v_weight = qkv_weight[-self.dim:, :]

        # Store K weight and bias (if any)
        self.register_buffer('k_weight', k_weight)
        if qkv.bias is not None:
            qkv_bias = qkv.bias.detach() # Use detach()
            self.register_buffer('q_bias', qkv_bias[:self.dim])
            self.register_buffer('k_bias', qkv_bias[self.dim:2*self.dim])
            self.register_buffer('v_bias', qkv_bias[-self.dim:])
        else:
            self.register_buffer('q_bias', None)
            self.register_buffer('k_bias', None)
            self.register_buffer('v_bias', None)

        # Compute SVD for Q and V (for LoRA initialization)
        U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
        U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)

        # Create low-rank approximations using top-r singular values/vectors
        U_q_r = U_q[:, :lora_r]
        S_q_r = S_q[:lora_r]
        Vh_q_r = Vh_q[:lora_r, :]
        q_low_rank = U_q_r @ torch.diag(S_q_r) @ Vh_q_r

        U_v_r = U_v[:, :lora_r]
        S_v_r = S_v[:lora_r]
        Vh_v_r = Vh_v[:lora_r, :]
        v_low_rank = U_v_r @ torch.diag(S_v_r) @ Vh_v_r

        # Calculate residual weights
        q_residual = q_weight - q_low_rank
        v_residual = v_weight - v_low_rank

        # SVD of the residuals
        U_q_res, S_q_res, Vh_q_res = torch.linalg.svd(q_residual, full_matrices=False)
        U_v_res, S_v_res, Vh_v_res = torch.linalg.svd(v_residual, full_matrices=False)

        # Store residual SVD components: U, Vh as buffers, S as learnable parameter
        self.register_buffer('U_q_res', U_q_res)
        self.register_buffer('Vh_q_res', Vh_q_res)
        self.S_q_res = nn.Parameter(S_q_res) # Learnable Sigma for Q residual

        self.register_buffer('U_v_res', U_v_res)
        self.register_buffer('Vh_v_res', Vh_v_res)
        self.S_v_res = nn.Parameter(S_v_res) # Learnable Sigma for V residual

        # Initialize LoRA linear layers using principal components
        self.linear_a_q = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_q = nn.Linear(lora_r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, lora_r, bias=False)
        self.linear_b_v = nn.Linear(lora_r, self.dim, bias=False)

        with torch.no_grad():
            self.linear_a_q.weight.copy_(torch.diag(torch.sqrt(S_q_r)) @ Vh_q_r)
            self.linear_b_q.weight.copy_(U_q_r @ torch.diag(torch.sqrt(S_q_r)))

            self.linear_a_v.weight.copy_(torch.diag(torch.sqrt(S_v_r)) @ Vh_v_r)
            self.linear_b_v.weight.copy_(U_v_r @ torch.diag(torch.sqrt(S_v_r)))

        # Ensure LoRA layers are trainable, while buffers are not
        self.S_q_res.requires_grad_(True)
        self.S_v_res.requires_grad_(True)


    def forward(self, x) -> torch.Tensor:

        # Compute LoRA outputs
        lora_q = self.linear_b_q(self.linear_a_q(x))
        lora_v = self.linear_b_v(self.linear_a_v(x))

        # Reconstruct residual Q and V weights using learnable singular values
        # Buffers and parameters should already be on the correct device/dtype
        q_res_weight_learnable = (self.U_q_res @
                                  torch.diag(self.S_q_res) @
                                  self.Vh_q_res)
        v_res_weight_learnable = (self.U_v_res @
                                  torch.diag(self.S_v_res) @
                                  self.Vh_v_res)


        # Compute output using reconstructed residual weights and stored bias
        q_res_output = F.linear(x, q_res_weight_learnable, self.q_bias)
        v_res_output = F.linear(x, v_res_weight_learnable, self.v_bias)

        # Compute K output using stored K weight/bias
        k_output = F.linear(x, self.k_weight, self.k_bias)

        # Combine residual output with LoRA output
        q_new = q_res_output + lora_q
        v_new = v_res_output + lora_v
        k_new = k_output # K remains unchanged

        return torch.cat((q_new, k_new, v_new), dim=-1)
