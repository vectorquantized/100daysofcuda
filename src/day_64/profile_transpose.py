import torch
import cblas


M, N = 1024, 1024

A = torch.randn(M, N, device="cuda", dtype=torch.float32)

for _ in range(10):
    _ = cblas.transpose_slow(A)
    _ = A.T.contiguous()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("cblas"):
        B_custom = cblas.transpose_slow(A)

    with torch.profiler.record_function("torch_transpose"):
        B_ref = A.T.contiguous()

print(f"Results match: {torch.allclose(B_custom, B_ref)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))