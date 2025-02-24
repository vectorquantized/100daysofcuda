import torch
import torch.nn.functional as F
import sigp_batched_in_out_channel as convolution

B, C_out, C_in, H, W = 16, 64, 3, 1024, 1024
kernel_size = (3, 3)

weight = torch.randn(C_out, C_in, *kernel_size, device="cuda", dtype=torch.float32)
input = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
output = torch.zeros((B, C_out, H - kernel_size[0] + 1, W - kernel_size[1] + 1), device=input.device, dtype=input.dtype)

print(f"{input.shape=}, {weight.shape=}")

for _ in range(10):
    _ = convolution.conv2D(input, weight, output)
    _ = F.conv2d(input, weight)

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
        C_ref = F.conv2d(input, weight).squeeze(1)
        print(f"{C_ref.shape=}")

print(f"Results match: {torch.allclose(C_custom, C_ref)}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# print(f"{C_ref=}")
# print("*" * 80)
# print(f"{C_custom=}")

diff = (C_custom - C_ref).abs().max()
print("Max diff =", diff.item())