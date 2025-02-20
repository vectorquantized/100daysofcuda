import torch
import torch.nn.functional as F
import sigp as convolution

B, H, W = 32, 1024, 1024
kernel_size = (3, 3)

weight = torch.randn(*kernel_size, device="cuda", dtype=torch.float32)
input = torch.randn(B, H, W, device="cuda", dtype=torch.float32)
output = torch.zeros_like(input, device=input.device, dtype=input.dtype)

print(f"{input.shape=}, {weight.shape=}")

for _ in range(10):
    _ = convolution.conv2D(input, weight, output)
    _ = F.conv2d(input[:, None, ...], weight[None, None, ...])

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_conv2d"):
        C_custom = convolution.conv2D(input, weight, output)

    with torch.profiler.record_function("torch_conv2d"):
        C_ref = F.conv2d(input[:, None, ...], weight[None, None, ...])

print(f"Results match: {torch.allclose(C_custom, C_ref.squeeze())}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# print(f"{C_ref.squeeze()=}")
# print("*" * 80)
# print(f"{C_custom=}")