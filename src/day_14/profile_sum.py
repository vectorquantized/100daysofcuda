import torch
import reduction
from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs/add_atomic_profiling"
writer = SummaryWriter(log_dir)

size = 1 << 30

A = torch.randn(size, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
) as prof:

    with torch.profiler.record_function("custom_sum"):
        C_custom = reduction.sum(A)
        torch.cuda.synchronize()

    with torch.profiler.record_function("torch sum"):
        C_ref = torch.sum(A)
        torch.cuda.synchronize()

print(f"Results match: {torch.allclose(C_custom, C_ref)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

writer.close()