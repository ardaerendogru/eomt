import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from analyze_relevance import analyze_singular_direction_relevance
from informed_pissa import InformedPiSSA, create_pissa_with_precomputed_relevance


def create_simple_model():
    """Create a simple model with a QKV layer for demonstration purposes"""
    hidden_dim = 64
    
    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
            
        def forward(self, x):
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            # Simple attention calculation (just for demo purposes)
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim)
            attn = torch.softmax(attn, dim=-1)
            output = torch.matmul(attn, v)
            return output
            
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, hidden_dim)
            self.attention = SimpleAttention()
            self.fc = nn.Linear(hidden_dim, 10)  # 10 classes for classification
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.attention(x)
            # Take the mean over the sequence dimension
            x = x.mean(dim=1)
            x = self.fc(x)
            return x
    
    return SimpleModel()


def create_dummy_data():
    """Create dummy data for demonstration"""
    # Create random token IDs (vocabulary size 100)
    inputs = torch.randint(0, 100, (32, 10))  # batch_size=32, seq_len=10
    labels = torch.randint(0, 10, (32,))  # 10 classes
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=4)
    
    return dataloader


def visualize_relevance(results, title='Singular Direction Relevance'):
    """Visualize relevance scores"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Q relevance
    q_indices = results['q_indices']
    q_relevance = results['q_relevance']
    q_singular = results['q_singular_values'][q_indices]
    
    axs[0].bar(range(len(q_indices)), q_relevance)
    axs[0].set_title('Q Matrix - Gradient Relevance Scores (Top Indices)')
    axs[0].set_xlabel('Index Rank')
    axs[0].set_ylabel('Relevance Score')
    
    # Add singular values as a line plot on secondary y-axis
    ax2 = axs[0].twinx()
    ax2.plot(range(len(q_indices)), q_singular, 'r-', marker='o')
    ax2.set_ylabel('Singular Value', color='r')
    
    # Plot V relevance
    v_indices = results['v_indices']
    v_relevance = results['v_relevance']
    v_singular = results['v_singular_values'][v_indices]
    
    axs[1].bar(range(len(v_indices)), v_relevance)
    axs[1].set_title('V Matrix - Gradient Relevance Scores (Top Indices)')
    axs[1].set_xlabel('Index Rank')
    axs[1].set_ylabel('Relevance Score')
    
    # Add singular values as a line plot on secondary y-axis
    ax2 = axs[1].twinx()
    ax2.plot(range(len(v_indices)), v_singular, 'r-', marker='o')
    ax2.set_ylabel('Singular Value', color='r')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Visualization saved as {title.replace(' ', '_')}.png")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and data
    model = create_simple_model()
    data_loader = create_dummy_data()
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Get QKV layer
    qkv_layer = model.attention.qkv
    
    # Run gradient relevance analysis
    print("Analyzing gradient relevance for singular directions...")
    results = analyze_singular_direction_relevance(
        model=model,
        qkv_layer=qkv_layer,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device='cpu',  # Use CPU for this demo
        batch_limit=5,
        output_dir='./'
    )
    
    # Visualize relevance scores
    visualize_relevance(results)
    
    # Create InformedPiSSA
    lora_r = 8  # Number of rank dimensions to use
    
    # Method 1: Direct integration with gradient analysis
    print("\nCreating InformedPiSSA with direct gradient analysis...")
    pissa_direct = InformedPiSSA(
        qkv=qkv_layer,
        lora_r=lora_r,
        model=model,
        data_loader=data_loader,
        loss_fn=loss_fn,
        device='cpu',
        batch_limit=5
    )
    
    # Method 2: Using pre-computed relevance scores
    print("\nCreating InformedPiSSA with pre-computed relevance scores...")
    pissa_precomputed = create_pissa_with_precomputed_relevance(
        qkv=qkv_layer,
        lora_r=lora_r,
        q_relevance_file='q_relevance_indices.txt',
        v_relevance_file='v_relevance_indices.txt'
    )
    
    # Compare parameter counts
    total_params_original = sum(p.numel() for p in qkv_layer.parameters())
    total_params_pissa = sum(p.numel() for p in pissa_direct.parameters())
    
    print(f"\nParameter count comparison:")
    print(f"Original QKV layer: {total_params_original}")
    print(f"InformedPiSSA: {total_params_pissa}")
    print(f"Parameter reduction: {total_params_original / total_params_pissa:.2f}x")
    
    # Test forward pass
    dummy_input = torch.randn(4, 64)  # batch_size=4, hidden_dim=64
    
    with torch.no_grad():
        qkv_output = qkv_layer(dummy_input)
        pissa_output = pissa_direct(dummy_input)
        
    print(f"\nOutput shape comparison:")
    print(f"Original QKV output shape: {qkv_output.shape}")
    print(f"InformedPiSSA output shape: {pissa_output.shape}")
    
    # Compute error
    with torch.no_grad():
        error = torch.norm(qkv_output - pissa_output) / torch.norm(qkv_output)
        print(f"Relative error: {error.item():.6f}")


if __name__ == "__main__":
    main() 