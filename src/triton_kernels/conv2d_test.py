
import torch
import torch.nn.functional as F
import pytest

# Import the Conv2D module from your project.
# Adjust the import path according to your project structure.
from triton_kernels.conv2d import Conv2D

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_conv2d_output():
    with torch.no_grad():
        num_batch, in_channels, height, width = 16, 3, 512, 512     # Input dimensions.
        out_channels, kH, kW = 1, 3, 3                # Kernel dimensions.
        stride = 1
        # We're assuming cuda device is available, there's no need to guard this as we want to run on CUDA devices for now.
        # we default to fp32 for now, low-bit kernels will be added later.
        input_tensor = torch.randn(num_batch, in_channels, height, width, device='cuda', dtype=torch.float32)
        conv2d = Conv2D(in_channels, out_channels, kernel_size=(kH, kW), stride=stride, device=input_tensor.device, dtype=input_tensor.dtype)
        output_tensor = conv2d(input_tensor)
        torch_output = F.conv2d(input_tensor, conv2d.weight)
        if torch.allclose(output_tensor, torch_output, rtol=1e-3, atol=1e-3):
            print("✅ Triton and Torch conv2d implementations match")
        else:
            print("❌ Triton and Torch conv2d implementations differ")