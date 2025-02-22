#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/convolution.h"
#include "../cuda/cuda_utils.h"
#include "batched_conv2d.h"

torch::Tensor batched_conv2d_forward(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().is_cuda(), "Kernel must be a CUDA tensor");
    TORCH_CHECK(input.ndimension() == 4, "Input must be 4D tensor (B,C,H,W)");
    TORCH_CHECK(kernel.ndimension() == 3, "Kernel must be 3D tensor (C,Kh,Kw)");
    TORCH_CHECK(input.size(1) == kernel.size(0), "Channel dimensions must match");
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int filter_size = kernel.size(1);
    int filter_radius = (filter_size - 1) / 2;

    int out_height = height - 2 * filter_radius;
    int out_width = width - 2 * filter_radius;

    // Use fixed block dimensions
    int in_tile_width = OUT_TILE_WIDTH + filter_size - 1;
    dim3 block_dim(in_tile_width, in_tile_width);
    dim3 grid_dim(
        CEIL_DIV(out_width, OUT_TILE_WIDTH),
        CEIL_DIV(out_height, OUT_TILE_WIDTH),
        batch_size
    );
    
    size_t shared_mem_size = sizeof(float) * channels * in_tile_width * in_tile_width;
    
    conv2d_tiled_channel<float><<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        filter_size, filter_radius,
        batch_size, channels,
        height, width,
        out_height, out_width
    );
    
    return output;
}