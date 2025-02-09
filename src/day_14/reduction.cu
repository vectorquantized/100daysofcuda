#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/reduction.h"

torch::Tensor sum_atomic_forward(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda() , "A must be a cuda tensor");

    int size = input.size(0);
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    size_t shared_memory_size = threads_per_block * sizeof(float);
    auto partial = torch::zeros({blocks_per_grid}, torch::TensorOptions().device(input.device()).dtype(input.dtype()));
    auto result = torch::zeros({1}, torch::TensorOptions().device(input.device()).dtype(input.dtype()));
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(blocks_per_grid);
    
    sum_atomic<float><<<grid_dim, block_dim, shared_memory_size>>>(input.data_ptr<float>(), partial.data_ptr<float>(), 
                                                                   result.data_ptr<float>(), size);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", &sum_atomic_forward, "Sum Atomic forward pass");
}
