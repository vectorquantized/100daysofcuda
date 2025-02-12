import torch
import torch.nn.functional as F
import convolution

B, C, H, W = 1, 3, 128, 128
kernel_size = (3,3)
weight = torch.randn(C, *kernel_size, device="cuda", dtype=torch.float32)
input = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)

print(f"{input.shape=}, {weight.shape=}")

for _ in range(10):
    _ = convolution.conv2D(input, weight)
    _ = F.conv2d(input, weight.unsqueeze(0))

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_conv2d"):
        C_custom = convolution.conv2D(input, weight)

    with torch.profiler.record_function("torch_conv2d"):
        C_ref = F.conv2d(input, weight.unsqueeze(0))

print(f"Results match: {torch.allclose(C_custom, C_ref)}")

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"{C_ref=}")
print("*" * 80)
print(f"{C_custom=}")