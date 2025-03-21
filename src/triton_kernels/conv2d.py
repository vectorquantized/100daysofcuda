
from triton_kernels.base import Kernel
from typing import Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from triton_kernels.functional.kernels.convolution import conv2d_kernel
from typing import Callable, Optional, Union
import triton
import math

InitFuncType = Callable[[torch.Tensor, Optional[float]], torch.Tensor]

class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, tuple[int, int]], stride: int,
                 device: Union[torch.device, str] = "cuda",
                 dtype: torch.dtype = torch.float32,
                 init_func: InitFuncType = init.kaiming_uniform_,
                 bias = False):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "Kernel size should either be a tuple of length 2 or of type int."
            self.kH, self.kW = kernel_size
        else:
            assert isinstance(kernel_size, int), "Kernel size is neither int nor a tuple of ints."
            self.kH, self.kW = kernel_size, kernel_size
        self.device = device
        self.dtype = dtype
        self.init_func = init_func 
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kH, self.kW, device=self.device, dtype=self.dtype), requires_grad=True)
        self.bias = None
        self.bias = None if not bias else nn.Parameter(torch.empty(self.out_channels, device=self.device, dtype=self.dtype), requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.init_func(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = input_tensor.shape
        assert self.in_channels == C, f"input's channel dimension should match kernel's channel dimensions, expected: {self.in_channels}, got: {C}"
        kH = self.kH
        kW = self.kW
        out_channels = self.out_channels
        stride = self.stride
        
        oH = (H - kH) // stride + 1
        oW = (W - kW) // stride + 1
        output_tensor = torch.empty(B, out_channels, oH, oW, device=input_tensor.device, dtype=input_tensor.dtype)
        
        def grid_fn(meta):
            # Extract the BLOCK_SIZE from the current configuration
            block_size = meta['BLOCK_SIZE']
            out_tile_height = block_size - kH + 1
            out_tile_width = block_size - kW + 1
            
            # Ensure tiles are at least 1x1
            out_tile_height = max(1, out_tile_height)
            out_tile_width = max(1, out_tile_width)
            
            num_tile_rows = (oH + out_tile_height - 1) // out_tile_height
            num_tile_cols = (oW + out_tile_width - 1) // out_tile_width
            
            return (B * out_channels, num_tile_rows, num_tile_cols)
        
        grid = grid_fn
        max_kH, max_kW = triton.next_power_of_2(kH), triton.next_power_of_2(kW)
    
        conv2d_kernel[grid](input_tensor, self.weight, output_tensor,
                        input_tensor.stride(0), input_tensor.stride(1), output_tensor.stride(0), output_tensor.stride(1), 
                        self.in_channels, out_channels, H, W,
                        self.weight.stride(0), self.weight.stride(1), self.stride, kH, kW, max_kH, max_kW)
    
    
        return output_tensor