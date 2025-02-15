#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"

#define TILE_WIDTH 16

torch::Tensor swiglu_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");

    int batch_size = A.size(0);
    int L = A.size(1);
    int D = A.size(2);

    auto C = torch::zeros({batch_size, L, D}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (L + TILE_WIDTH - 1) / TILE_WIDTH,
        (L + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );
    
    batched_gemm_tiled_ABT<float, TILE_WIDTH><<<grid_dim, block_dim>>>(A.data_ptr<float>(), 
                                                                       B.data_ptr<float>(), 
                                                                       C.data_ptr<float>(), 
                                                                       batch_size, L, D, scale);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swiglu_forward, "Swiglu CUDA forward pass");
}
