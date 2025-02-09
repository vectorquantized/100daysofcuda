#include <vector>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/normalization.h"
#include "../cuda/cuda_utils.h"


torch::Tensor layernorm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "beta must be a CUDA tensor");

    const int num_samples = input.size(0);
    const int num_features = input.size(1);

    int threads_per_block = 256;

    dim3 block_dim(threads_per_block);
    dim3 grid_dim(num_samples);

    // For the tiled kernel we need shared memory for 2 * threads_per_block floats.
    size_t shared_memory_size = 2 * threads_per_block * sizeof(float);

    auto output = torch::empty_like(input);

    layer_norm<<<grid_dim, block_dim, shared_memory_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        num_features,
        eps);

    // We've not settled on a post launch check mechanism yet,
    // I've been using CUDA_ERROR_CHECK, but TORCH_CHECK works too
    // just need one for consistency.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layernorm, "LayerNorm CUDA");
}