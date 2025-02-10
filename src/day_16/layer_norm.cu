#include <vector>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/normalization.h"
#include "../cuda/cuda_utils.h"
#define NUM_THREADS 1024


torch::Tensor layernorm(torch::Tensor input, at::IntArrayRef normalized_shape, torch::Tensor gamma, torch::Tensor beta, float eps) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1 + normalized_shape.size(),
                "Input tensor has ", input.dim(), " dimensions, but normalized_shape has ",
                normalized_shape.size(), " dimensions.");
    for (int i = 0; i < normalized_shape.size(); ++i) {
        TORCH_CHECK(input.size(i + 1) == normalized_shape[i],
                    "Mismatch at dimension ", i, ": expected ", normalized_shape[i],
                    " but got ", input.size(i + 1));
    }

    TORCH_CHECK(gamma.sizes().size() == normalized_shape.size(), "Size of gamma must match that of normalized_shape. Expected: ",
        normalized_shape.size(), " got: ", gamma.sizes().size());

    TORCH_CHECK(beta.sizes().size() == normalized_shape.size(), "Size of beta must match that of normalized_shape. Expected: ",
        normalized_shape.size(), " got: ", beta.sizes().size());

    for (int i = 0; i < normalized_shape.size(); ++i) {
        TORCH_CHECK(gamma.size(i) == normalized_shape[i],
                    "Mismatch between gamma and normalized_shape at dimension ", i, ": expected ", normalized_shape[i],
                    " but got ", gamma.size(i));
        TORCH_CHECK(beta.size(i) == normalized_shape[i],
        "Mismatch between beta and normalized_shape at dimension ", i, ": expected ", normalized_shape[i],
        " but got ", beta.size(i));
    }

    auto compute_num_features = [&normalized_shape]() -> int {
        int num_features = 1;
        for (size_t i = 0; i < normalized_shape.size(); ++i) {
            num_features *= normalized_shape[i];
        }
        return num_features;
    };

    const int num_samples = input.size(0);
    const int num_features = compute_num_features();

    int threads_per_block = std::min(NUM_THREADS, num_features);
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layernorm, "LayerNorm CUDA");
}