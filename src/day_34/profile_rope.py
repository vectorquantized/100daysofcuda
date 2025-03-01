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

# Create test data on CPU then move to GPU
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32)

# Create output tensors with same shape
q_out = torch.zeros_like(q)
k_out = torch.zeros_like(k)

# Generate position IDs
position_ids = torch.arange(0, seq_len, device="cuda", dtype=torch.int32) # L
pos_ids = position_ids[None, :].expand(batch_size, -1) # B, L

# Generate sine/cosine embeddings
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
t = torch.arange(0, seq_len, device="cuda", dtype=torch.float32)
freqs = torch.einsum("i,j->ij", t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()[:, :head_dim//2]  # [seq_len, head_dim/2]
sin = emb.sin()[:, :head_dim//2]  # [seq_len, head_dim/2]

print(f"Query shape: {q.shape}")
print(f"Key shape: {k.shape}")
print(f"t shape: {t.shape}")
print(f"freqs shape: {freqs.shape}")
print(f"Pos IDs shape: {pos_ids.shape}")
print(f"Cosine table shape: {cos.shape}")
print(f"Sine table shape: {sin.shape}")

def apply_rotary_pos_emb_torch(q, k, cos, sin):
    """Efficient PyTorch implementation of RoPE"""
    # Get shapes
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Reshape for broadcasting
    # [batch_size, seq_len, 1, head_dim/2] -> [batch_size, seq_len, num_heads, head_dim/2]
    cos_pos = cos[None, :, None, :].expand(batch_size, -1, 1, -1)
    sin_pos = sin[None, :, None, :].expand(batch_size, -1, 1, -1)
    
    # Process even and odd dimensions
    # Reshape q and k to separate even and odd dimensions
    q_even = q[:, :, :, 0::2]
    q_odd = q[:, :, :, 1::2]
    k_even = k[:, :, :, 0::2]
    k_odd = k[:, :, :, 1::2]
    
    print(f"{q_even.shape=}")
    print(f"{cos_pos.shape=}")
    # Apply rotation using broadcasting
    q_embed_even = q_even * cos_pos - q_odd * sin_pos
    q_embed_odd = q_odd * cos_pos + q_even * sin_pos
    k_embed_even = k_even * cos_pos - k_odd * sin_pos
    k_embed_odd = k_odd * cos_pos + k_even * sin_pos
    
    # Interleave the results
    q_embed = torch.zeros_like(q)
    k_embed = torch.zeros_like(k)
    
    q_embed[:, :, :, 0::2] = q_embed_even
    q_embed[:, :, :, 1::2] = q_embed_odd
    k_embed[:, :, :, 0::2] = k_embed_even
    k_embed[:, :, :, 1::2] = k_embed_odd
    
    return q_embed, k_embed

# Make sure tensors are in the right format for CUDA kernel
q_cuda = q.clone()  # [batch, seq, head, dim]
k_cuda = k.clone()

# For PyTorch reference, match the expected format
q_torch = q.clone() #q.permute(0, 2, 1, 3).contiguous()  # [batch, head, seq, dim]
k_torch = k.clone() #k.permute(0, 2, 1, 3).contiguous()

# Warmup
for _ in range(5):
    rope.apply_rope(q_cuda, k_cuda, pos_ids, cos, sin, q_out, k_out)
    q_ref, k_ref = apply_rotary_pos_emb_torch(q_torch, k_torch, cos, sin)
    torch.cuda.synchronize()

# q_ref = q_ref_perm.permute(0, 2, 1, 3).contiguous()  # Convert back to [B,S,H,D]
# k_ref = k_ref_perm.permute(0, 2, 1, 3).contiguous()
# Test custom CUDA implementation
start = time.time()
for _ in range(10):
    rope.apply_rope(q_cuda, k_cuda, pos_ids, cos, sin, q_out, k_out)
    torch.cuda.synchronize()
cuda_time = (time.time() - start) / 10
print(f"Custom CUDA implementation: {cuda_time*1000:.2f} ms")

# Test PyTorch implementation
start = time.time()
for _ in range(10):
    q_ref, k_ref = apply_rotary_pos_emb_torch(q_torch, k_torch, cos, sin)
    torch.cuda.synchronize()

# q_ref = q_ref_perm.permute(0, 2, 1, 3).contiguous()  # Convert back to [B,S,H,D]
# k_ref = k_ref_perm.permute(0, 2, 1, 3).contiguous()
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
        q_ref, k_ref = apply_rotary_pos_emb_torch(q_torch, k_torch, cos, sin)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# print(f"{q_ref=}")
# print("*"*80)
# print(f"{q_out=}")
# print("*"*80)
# Save results to visualize
if q_diff > 1e-5 or k_diff > 1e-5:
    # Save a small sample for debugging
    sample_idx = (0, 1, 0)  # batch 0, seq 0, head 0
    sample_q_custom = q_out[sample_idx].cpu().numpy()
    sample_q_torch = q_ref[sample_idx].cpu().numpy()
    sample_k_custom = k_out[sample_idx].cpu().numpy()
    sample_k_torch = k_ref[sample_idx].cpu().numpy()
    
    print("\nSample comparison for debugging:")
    print("Custom Q:", sample_q_custom)
    print("Torch Q:", sample_q_torch)
    print("Custom K:", sample_k_custom)
    print("Torch K:", sample_k_torch)


# Check specific values at a position
b, s, h = 0, 1, 0  # First batch, sequence, head
d_pair = 7  # First dimension pair

# Get the original values
q_even_orig = q[b, s, h, d_pair*2].item()
q_odd_orig = q[b, s, h, d_pair*2+1].item()

# Get position ID
pos_id = pos_ids[b, s].item()

# Get cos/sin values
cos_val = cos[pos_id, d_pair].item()
sin_val = sin[pos_id, d_pair].item()

# Calculate expected results manually
q_even_expected = q_even_orig * cos_val - q_odd_orig * sin_val
q_odd_expected = q_odd_orig * cos_val + q_even_orig * sin_val

# Get actual results
q_even_cuda = q_out[b, s, h, d_pair*2].item()
q_odd_cuda = q_out[b, s, h, d_pair*2+1].item()
q_even_torch = q_ref[b, s, h, d_pair*2].item()
q_odd_torch = q_ref[b, s, h, d_pair*2+1].item()

print(f"Original values: q_even={q_even_orig}, q_odd={q_odd_orig}")
print(f"Position ID: {pos_id}, cos={cos_val}, sin={sin_val}")
print(f"Expected: q_even={q_even_expected}, q_odd={q_odd_expected}")
print(f"CUDA: q_even={q_even_cuda}, q_odd={q_odd_cuda}")
print(f"PyTorch: q_even={q_even_torch}, q_odd={q_odd_torch}")