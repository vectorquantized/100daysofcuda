
#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/convolution.h"
#include "../cuda/cuda_utils.h"

#define TILE_WIDTH 16

torch::Tensor batched_conv2d_forward(torch::Tensor input, torch::Tensor kernel) {
    TORCH_CHECK(input.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(kernel.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(input.ndimension() == 4, "input must be 4D tensors.");
    TORCH_CHECK(kernel.ndimension() == 3, "kernel must be 3D tensors.");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "kernel must be float32");

    int batch_size = input.size(0);
    int channels = input.size(1);
    int M = input.size(2);
    int N = input.size(3);
    int filter_size = kernel.size(1);
    int filter_radius = (filter_size  - 1) / 2;

    int out_rows = M - 2 * filter_radius;
    int out_cols = N - 2 * filter_radius;

    auto output = torch::zeros({batch_size, out_rows, out_cols}, torch::TensorOptions().device(input.device()).dtype(input.dtype()));

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (out_cols + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_rows + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );
    
    conv2D<float><<<grid_dim, block_dim>>>(input.data_ptr<float>(), 
                                       kernel.data_ptr<float>(), output.data_ptr<float>(), 
                                       filter_radius, batch_size, channels, M, N);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2D", &batched_conv2d_forward, "2D convolution CUDA forward pass");
}