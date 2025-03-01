import torch
import gemm_tiled_bench

batch_size, M, K, N = 8, 256, 512, 256 
scale = 1.0

# Create input tensors
A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float32)
B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)


for _ in range(10):
    _ = gemm_tiled_bench.bmm(A, B, scale)
    _ = torch.matmul(A, B)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_bmm"):
        C_custom = gemm_tiled_bench.bmm(A, B, scale)

    with torch.profiler.record_function("torch_bmm"):
        C_torch = torch.bmm(A, B)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))