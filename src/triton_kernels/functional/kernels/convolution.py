import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_stages=3, num_warps=1),
        # triton.Config({'BLOCK_SIZE': 16}, num_stages=3, num_warps=2),
        # triton.Config({'BLOCK_SIZE': 8}, num_stages=8, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 32}, num_stages=8, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE': 16}, num_stages=3, num_warps=8),
    ],
    key=['height', 'width', 'stride', 'kH', 'kW'],
)
@triton.jit
def conv2d_kernel(input_ptr: torch.Tensor, kernel_ptr: torch.Tensor, output: torch.Tensor, 
                  in_batch_stride: int, in_channel_stride: int, out_batch_stride: int, out_channel_stride: int, 
                  in_channels: int, out_channels: int, height: int, width: int, 
                  k_out_channel_stride: int, k_in_channel_stride: int, stride: int, kH: int, kW: int, 
                  max_kH: tl.constexpr, max_kW: tl.constexpr, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr): 
    
    batch_idx = tl.program_id(0) // out_channels
    out_c = tl.program_id(0) % out_channels
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)
    
    out_height = (height - kH) // stride + 1
    out_width = (width - kW) // stride + 1
    
    OUT_TILE_HEIGHT = BLOCK_SIZE - kH + 1
    OUT_TILE_WIDTH = BLOCK_SIZE - kW + 1
    
    out_tile_row = tile_row * OUT_TILE_HEIGHT
    out_tile_col = tile_col * OUT_TILE_WIDTH
    
    kernel_row_offset = tl.arange(0, max_kH)
    
    kernel_col_offset = tl.arange(0, max_kW)

    # Create a mask for valid kernel indices.
    kernel_mask = (kernel_row_offset[:, None] < kH) & (kernel_col_offset[None, :] < kW)
    
    for i in tl.range(0, OUT_TILE_HEIGHT, num_stages=num_stages):
        for j in tl.range(0, OUT_TILE_WIDTH, num_stages=num_stages):
        
            out_row = out_tile_row + i
            out_col = out_tile_col + j
            if out_row < out_height and out_col < out_width:
                acc = 0.0
                input_row = out_row * stride
                input_col = out_col * stride
                for c in tl.range(0, in_channels, num_stages=num_stages):
                    # Load a full kernel block using masked load.
                    kernel_block = tl.load(
                        kernel_ptr + out_c * k_out_channel_stride + c * k_in_channel_stride + kernel_row_offset[:, None] * kW + kernel_col_offset[None, :],
                        mask=kernel_mask, other=0.0
                    )
                    patch_ptr = input_ptr + batch_idx * in_batch_stride + c * in_channel_stride + input_row * width + input_col
                    input_patch = patch_ptr + kernel_row_offset[:, None] * width + kernel_col_offset[None, :]
                    mask = (input_row + kernel_row_offset[:, None] < height) & (input_col + kernel_col_offset[None, :] < width) & kernel_mask
                    input_block = tl.load(input_patch, mask = mask, other=0.0)
                    acc += tl.sum(input_block * kernel_block)
                tl.store(output + batch_idx * out_batch_stride + out_c * out_channel_stride + out_row * out_width + out_col, acc)
                        
                    
        