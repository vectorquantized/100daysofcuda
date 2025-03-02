import torch
import torch.nn.functional as F
import rope
import math
import time

# Define test dimensions
batch_size = 4
seq_len = 8192
num_heads = 16
head_dim = 1024
# Base frequency for RoPE. Default is 10000.0.
theta = 10000.0

# Create test data on CPU then move to GPU
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)

# Create output tensors with same shape
q_out = torch.zeros_like(q)
k_out = torch.zeros_like(k)

# Compute frequency tensor
dim = torch.arange(0, head_dim, 2, device=q.device).float()  # Shape: (D/2,)
freqs = 1.0 / (theta ** (dim / head_dim))  # Shape: (D/2,)

# Compute position indices
positions = torch.arange(seq_len, device=q.device).float()  # Shape: (L,)
pos_ids = positions[None, :].expand(batch_size, -1).int() # B, L
angles = torch.einsum("l,d->ld", positions, freqs)  # Shape: (L, D/2)

# Compute sin and cos
sin = torch.sin(angles)  # Shape: (L, D/2)
cos = torch.cos(angles)  # Shape: (L, D/2)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input tensor based on rotation matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, H, D).
        cos (torch.Tensor): real part of the embs, shape: (B, L, H, D/2)
        sin (torch.Tensor): imaginary part of the embs, shape: (B, L, H, D/2)

    Returns:
        torch.Tensor: The tensor after applying rotation.
    """
    
    # Select even and odd indices
    x_even = x[..., ::2]  # Shape: (B, L, H, D/2)
    x_odd = x[..., 1::2]  # Shape: (B, L, H, D/2)

    # Apply rotation
    x_rot = torch.empty_like(x)
    x_rot[..., ::2] = x_even * cos - x_odd * sin
    x_rot[..., 1::2] = x_odd * cos + x_even * sin

    return x_rot

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.

    Args:
        q (torch.Tensor): Query tensor of shape (B, L, H, D).
        k (torch.Tensor): Key tensor of shape (B, L, H, D).
        cos (torch.Tensor): real part of the embs, shape: (L, D/2)
        sin (torch.Tensor): imaginary part of the embs, shape: (L, D/2)

    Returns:
        torch.Tensor: The tensor after applying RoPE.
    """
    B, L, H, D = q.shape
    assert D % 2 == 0, "Head dimension must be even for RoPE."
    assert q.shape == k.shape, f"Query and key shapes should match, got: {q.shape=} and {k.shape=}"

    # Expand for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)
    cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)
    q_rot = rotate(q, cos, sin)
    k_rot = rotate(k, cos, sin)
    
    return q_rot, k_rot

# Warmup
for _ in range(5):
    rope.apply_rope(q, k, pos_ids, cos, sin, q_out, k_out)
    apply_rope(q, k, cos, sin)
    # torch.cuda.synchronize()

# Test custom CUDA implementation
start = time.time()
for _ in range(10):
    rope.apply_rope(q, k, pos_ids, cos, sin, q_out, k_out)
    # torch.cuda.synchronize()
cuda_time = (time.time() - start) / 10
print(f"Custom CUDA implementation: {cuda_time*1000000:.2f} us")

# Test PyTorch implementation
start = time.time()
for _ in range(10):
    q_ref, k_ref = apply_rope(q, k, cos, sin)
    # torch.cuda.synchronize()

torch_time = (time.time() - start) / 10
print(f"PyTorch implementation: {torch_time*1000000:.2f} us")
print(f"Speedup: {torch_time/cuda_time:.2f}x")

# Compare results
q_diff = (q_out - q_ref).abs().max().item()
k_diff = (k_out - k_ref).abs().max().item()
print(f"Maximum query difference: {q_diff}")
print(f"Maximum key difference: {k_diff}")
print(f"Results match: {q_diff < 1e-5 and k_diff < 1e-5}")

# Profile with torch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_rope"):
        rope.apply_rope(q, k, pos_ids, cos, sin, q_out, k_out)
        # torch.cuda.synchronize()
        
    with torch.profiler.record_function("torch_rope"):
        q_ref, k_ref = apply_rope(q, k, cos, sin)
        # torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))