import torch
import torch.nn.functional as F
import sigp as convolution

B, C_in, H, W = 1, 3, 16, 16
kernel_size = (3, 3)

weight = torch.randn(C_in, *kernel_size, device="cuda", dtype=torch.float32)
input = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
output = torch.zeros((B, H - kernel_size[0] + 1, W - kernel_size[1] + 1), device=input.device, dtype=input.dtype)

print(f"{input.shape=}, {weight.shape=}")

for _ in range(10):
    _ = convolution.conv2D(input, weight, output)
    _ = F.conv2d(input, weight[None, ...])

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True
) as prof:
    with torch.profiler.record_function("custom_conv2d"):
        C_custom = convolution.conv2D(input, weight, output)
        print(f"{C_custom.shape=}")
        

    with torch.profiler.record_function("torch_conv2d"):
        C_ref = F.conv2d(input, weight[None, ...]).squeeze(1)
        print(f"{C_ref.shape=}")

print(f"Results match: {torch.allclose(C_custom, C_ref)}")

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(f"{C_ref=}")
print("*" * 80)
print(f"{C_custom=}")

diff = (C_custom - C_ref).abs().max()
print("Max diff =", diff.item())