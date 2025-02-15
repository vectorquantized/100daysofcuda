import torch
import gemm_tiled_fused
from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs/gemm_profiling"
writer = SummaryWriter(log_dir)

batch_size, L, D = 8, 256, 512
scale = 1.0

A = torch.randn(batch_size, L, D, device="cuda", dtype=torch.float32)
B = torch.randn(batch_size, L, D, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
) as prof:

    with torch.profiler.record_function("custom_bmm_transpose_fused"):
        C_custom = gemm_tiled_fused.bmm(A, B, scale)
        torch.cuda.synchronize()

    with torch.profiler.record_function("torch_bmm_transpose"):
        C_ref = torch.bmm(A, B.transpose(1,2))
        torch.cuda.synchronize()

print(f"Matrices match: {torch.allclose(C_custom, C_ref,rtol=1e-3, atol=1e-3)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

writer.close()