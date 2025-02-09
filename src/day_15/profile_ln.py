import torch
import normalization
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

log_dir = "./logs/ln_profiling"
writer = SummaryWriter(log_dir)
B, D = 1024, 8192
A = torch.randn((B, D), device="cuda", dtype=torch.float32)
gamma = torch.ones(D, device="cuda", dtype=torch.float32)
beta  = torch.zeros(D, device="cuda", dtype=torch.float32)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
) as prof:

    with torch.profiler.record_function("custom_layer norm"):
        C_custom = normalization.layer_norm(A, gamma, beta, 1e-5)
        torch.cuda.synchronize()

    with torch.profiler.record_function("torch layer norm"):
        C_ref = F.layer_norm(A, normalized_shape=(D,), weight=gamma, bias=beta, eps=1e-5)
        torch.cuda.synchronize()

print(f"Results match: {torch.allclose(C_custom, C_ref)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

writer.close()