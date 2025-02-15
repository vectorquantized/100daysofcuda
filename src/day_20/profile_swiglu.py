import torch
import swiglu
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from dataclasses import dataclass

log_dir = "./logs/swiglu_profiling"
writer = SummaryWriter(log_dir)

batch_size, L, D = 8, 256, 512
scale = 1.0
multiplier = 1.0

A = torch.randn(batch_size, L, D, device="cuda", dtype=torch.float32)

@dataclass(kw_only=True)
class FeedForwardConfig:
    """FeedForward Config"""

    input_dim: int
    multiplier: int = 4
    hidden_dim: int = field(init=False)
    output_dim: int = field(init=False)

    def __post_init__(self):
        self.hidden_dim = int(self.multiplier * self.input_dim)
        self.output_dim = self.input_dim


class SwigLU(nn.Module):
    def __init__(self, cfg: FeedForwardConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=False)
        self.activation_fn = nn.SiLU()

    def forward( # pylint: disable=arguments-differ
         self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through feedForward Layer.

        Args:
            input_batch: a torch tensor of shape: (B, L, D)

        Returns:
            a torch tensor of shape: (B, L, D)
        """
        gated = self.gate_proj(input_batch)
        x = self.up_proj(input_batch)  # B, L, hidden_dim
        x = x * self.activation_fn(gated)  # B, L, hidden_dim
        x = self.down_proj(x)  # B, L, D
        return x

cfg = FeedForwardConfig(input_dim=D, multiplier = multiplier)
torch_swiglu = SwigLU(cfg)
print(f"{torch_swiglu=}")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
) as prof:

    with torch.profiler.record_function("custom_swiglu"):
        C_custom = swiglu.forward()
        torch.cuda.synchronize()

    with torch.profiler.record_function("torch_swiglu"):
        C_ref = torch_swiglu(A)
        torch.cuda.synchronize()

print(f"Matrices match: {torch.allclose(C_custom, C_ref,rtol=1e-3, atol=1e-3)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

writer.close()