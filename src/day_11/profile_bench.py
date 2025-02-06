import torch
import attention_bench  # Your CUDA extension

# Define matrix sizes
M, K, N = 512, 1024, 256  # Adjust size for benchmarking
scale = 1.0

# Create input tensors
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

# Ensure warm-up to eliminate cold start effects
for _ in range(10):
    _ = attention_bench.matmul(A, B, scale)
    _ = torch.matmul(A, B)

# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_gemm_tiled"):
        C_custom = attention_bench.matmul(A, B, scale)

    with torch.profiler.record_function("torch_matmul"):
        C_torch = torch.matmul(A, B)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))