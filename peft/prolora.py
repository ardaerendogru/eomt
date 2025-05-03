import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Define the ProLoRA Linear Layer
class ProLoRA(nn.Module):
    """
    A Linear layer wrapped with ProLoRA (Parameter-Efficient Re-Tuning).

    Applies element-wise modulation W_finetuned = W_orig * (1 + AB^T),
    where W_orig is the frozen pre-trained weight, and A, B are trainable
    low-rank matrices. Relies on PyTorch Lightning or similar framework for device placement.
    """
    def __init__(self, original_layer: nn.Linear, lora_r: int):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = lora_r
        self.has_bias = original_layer.bias is not None

        # Store original weight and bias (frozen).
        # Let Lightning handle moving these to the correct device later via model.to(device).
        # Register them as buffers so they are correctly moved and saved.
        self.register_buffer('original_weight', original_layer.weight.data.clone().detach().requires_grad_(False))
        if self.has_bias:
            self.register_buffer('original_bias', original_layer.bias.data.clone().detach().requires_grad_(False))
        else:
            # Explicitly register None buffer if no bias, though usually handled.
            # Or just don't store it and check self.has_bias in forward. Let's keep the check simple.
            self.register_buffer('original_bias', None, persistent=False) # Use persistent=False for None

        # Define trainable low-rank matrices A and B.
        # Initialize them on the default device (usually CPU). Lightning will move them.
        # A: (out_features, rank)
        # B: (in_features, rank)
        self.spert_A = nn.Parameter(torch.empty(self.out_features, self.rank)) # Default device/dtype
        self.spert_B = nn.Parameter(torch.empty(self.in_features, self.rank)) # Default device/dtype

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with zeros so (1 + AB^T) starts as 1
        nn.init.zeros_(self.spert_A)
        # Initialize B similarly to typical weight initializations
        nn.init.kaiming_uniform_(self.spert_B, a=math.sqrt(5))
        # Initialization happens on the device they currently reside on (initially default device).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- REMOVED DEVICE CHECK FOR x ---
        # PyTorch Lightning is responsible for ensuring x is on the correct device.
        # expected_device = self.original_weight.device # No longer needed
        # if x.device != expected_device:              # No longer needed
        #      x = x.to(expected_device)               # No longer needed

        # Get the bias term - handle None case
        bias = self.original_bias if self.has_bias else None

        # Compute the low-rank update matrix: delta_M = A @ B^T
        # This computation now happens on the device Lightning placed the module on.
        delta_M = self.spert_A @ self.spert_B.T

        # Compute the full modulation mask: M = 1 + delta_M
        modulation_mask = 1.0 + delta_M

        # Compute the effective finetuned weight: W_finetuned = W_orig * M (element-wise)
        # Operation happens on the correct device.
        w_finetuned = self.original_weight * modulation_mask

        # Perform the standard matrix multiplication with input x
        # All tensors (x, w_finetuned, bias) should be on the same device (managed by Lightning).
        output = F.linear(x, w_finetuned, bias)

        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, rank={}, has_bias={}'.format(
            self.in_features, self.out_features, self.rank, self.has_bias
        )

    def get_effective_weight(self) -> torch.Tensor:
        """Returns the computed effective weight matrix."""
        # Calculation happens on the module's current device.
        delta_M = self.spert_A @ self.spert_B.T
        modulation_mask = 1.0 + delta_M
        return self.original_weight * modulation_mask

    def merge(self) -> nn.Linear:
        """
        Computes the final effective weight matrix and returns a standard
        nn.Linear layer with this weight and the original bias.
        The new layer will be on the same device as this module.
        """
        effective_weight = self.get_effective_weight() # On the module's device

        # Create the new layer directly on the target device and with the correct dtype
        target_device = effective_weight.device
        target_dtype = effective_weight.dtype
        merged_layer = nn.Linear(self.in_features, self.out_features, bias=self.has_bias,
                                 device=target_device, dtype=target_dtype)

        # Use context manager to avoid tracking gradients during data assignment
        with torch.no_grad():
            merged_layer.weight.copy_(effective_weight)
            if self.has_bias:
                # Ensure original_bias is also on the correct device before copying
                bias_to_copy = self.original_bias.to(device=target_device, dtype=target_dtype)
                merged_layer.bias.copy_(bias_to_copy)

        return merged_layer