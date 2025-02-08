import torch
import gemm_tiled_bench
from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs/gemm_profiling"
writer = SummaryWriter(log_dir)

batch_size, M, K, N = 8, 8192, 16384, 4096
scale = 1.0

A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float32)
B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
) as prof:

    with torch.profiler.record_function("custom_bmm"):
        C_custom = gemm_tiled_bench.bmm(A, B, scale)
        torch.cuda.synchronize()

    with torch.profiler.record_function("custom_bmm_coarsened"):
        C_custom_coarsened = gemm_tiled_bench.bmm_coarsened(A, B, scale)
        torch.cuda.synchronize()

print(f"Matrices match: {torch.allclose(C_custom, C_custom_coarsened)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

writer.close()