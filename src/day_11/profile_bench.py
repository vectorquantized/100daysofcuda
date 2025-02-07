import torch
import gemm_tiled_bench

M, K, N = 512, 1024, 256 
scale = 1.0

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)


for _ in range(10):
    _ = gemm_tiled_bench.matmul(A, B, scale)
    _ = torch.matmul(A, B)

# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_gemm_tiled"):
        C_custom = gemm_tiled_bench.matmul(A, B, scale)

    with torch.profiler.record_function("torch_matmul"):
        C_torch = torch.matmul(A, B)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))