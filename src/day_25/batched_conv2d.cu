#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/convolution.h"
#include "../cuda/cuda_utils.h"
#include "batched_conv2d.h"

torch::Tensor batched_conv2d_forward(torch::Tensor input, torch::Tensor kernel, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(kernel.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(input.ndimension() == 3, "input must be 3D tensors.");
    TORCH_CHECK(input.ndimension() == output.ndimension(), "Input and Output must be of the same dimension");
    TORCH_CHECK(input.size(0) == output.size(0), "Batch dimension should be of the same size for both input and output.");
    TORCH_CHECK(kernel.ndimension() == 2, "kernel must be 2D tensors.");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "kernel must be float32");

    int batch_size = input.size(0);
    // int channels = input.size(1);
    int M = input.size(1);
    int N = input.size(2);
    // int out_channels = kernel.size(0);
    int filter_size = kernel.size(0);
    int filter_radius = (filter_size  - 1) / 2;

    int out_rows = M - 2 * filter_radius;
    int out_cols = N - 2 * filter_radius;

    

    int in_tile_width = OUT_TILE_WIDTH + filter_size - 1;
    dim3 block_dim(in_tile_width, in_tile_width);
    dim3 grid_dim(
        CEIL_DIV(out_cols, OUT_TILE_WIDTH),
        CEIL_DIV(out_rows, OUT_TILE_WIDTH),
        batch_size
    );
    int shared_mem_size = sizeof(float) * (in_tile_width * in_tile_width);
    conv2d_tiled<float><<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), 
        kernel.data_ptr<float>(), 
        output.data_ptr<float>(),
        filter_size, filter_radius,
        batch_size, M, N, out_cols, out_rows);
    return output;
}