import torch
import torch.nn.functional as F
import rope
import math
import time

# Define test dimensions
batch_size = 1
seq_len = 8
num_heads = 1
head_dim = 16
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
pos_ids = positions[None, :].expand(batch_size, -1) # B, L
angles = torch.einsum("l,d->ld", positions, freqs)  # Shape: (L, D/2)

# Compute sin and cos
sin = torch.sin(angles)  # Shape: (L, D/2)
cos = torch.cos(angles)  # Shape: (L, D/2)

print(f"Query shape: {q.shape}")
print(f"Key shape: {k.shape}")
# print(f"t shape: {t.shape}")
print(f"freqs shape: {freqs.shape}")
print(f"Pos IDs shape: {pos_ids.shape}")
print(f"Cosine table shape: {cos.shape}")
print(f"Sine table shape: {sin.shape}")

def apply_rope(x: torch.Tensor, cos, sin):
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, H, D).
        cos (torch.Tensor): real part of the embs, shape: (L, D/2)
        sin (torch.Tensor): imaginary part of the embs, shape: (L, D/2)

    Returns:
        torch.Tensor: The tensor after applying RoPE.
    """
    B, L, H, D = x.shape
    assert D % 2 == 0, "Head dimension must be even for RoPE."

    # Expand for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)
    cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)

    # Split into real and imaginary parts
    x1, x2 = x[..., :D//2], x[..., D//2:]  # Shape: (B, L, H, D/2)

    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return x_rot

# Warmup
for _ in range(5):
    rope.apply_rope(q, k, pos_ids, cos, sin, q_out, k_out)
    q_ref = apply_rope(q, cos, sin)
    k_ref = apply_rope(k, cos, sin)
    torch.cuda.synchronize()


# Test custom CUDA implementation
start = time.time()
for _ in range(10):
    rope.apply_rope(q, k, pos_ids, cos, sin, q_out, k_out)
    torch.cuda.synchronize()
cuda_time = (time.time() - start) / 10
print(f"Custom CUDA implementation: {cuda_time*1000:.2f} ms")

# Test PyTorch implementation
start = time.time()
for _ in range(10):
    q_ref = apply_rope(q, cos, sin)
    k_ref = apply_rope(k, cos, sin)
    torch.cuda.synchronize()

print(f"{q_ref.shape=}")
print(f"{q_out.shape=}")
torch_time = (time.time() - start) / 10
print(f"PyTorch implementation: {torch_time*1000:.2f} ms")
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
        torch.cuda.synchronize()
        
    with torch.profiler.record_function("torch_rope"):
        q_ref = apply_rope(q, cos, sin)
        k_ref = apply_rope(k, cos, sin)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# print(f"{q_ref=}")
# print("*"*80)
# print(f"{q_out=}")
# print("*"*80)
# Save results to visualize
# if q_diff > 1e-5 or k_diff > 1e-5:
#     # Save a small sample for debugging
#     sample_idx = (0, 1, 0)  # batch 0, seq 0, head 0
#     sample_q_custom = q_out[sample_idx].cpu().numpy()
#     sample_q_torch = q_ref[sample_idx].cpu().numpy()
#     sample_k_custom = k_out[sample_idx].cpu().numpy()
#     sample_k_torch = k_ref[sample_idx].cpu().numpy()
    
#     print("\nSample comparison for debugging:")
#     print("Custom Q:", sample_q_custom)
#     print("Torch Q:", sample_q_torch)
#     print("Custom K:", sample_k_custom)
#     print("Torch K:", sample_k_torch)