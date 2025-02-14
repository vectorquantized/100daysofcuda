#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"

#define TILE_WIDTH 16

torch::Tensor batched_gemm_tiled_ABt_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(B.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(A.ndimension() == 3 && B.ndimension() == 3 , "A and B must be 3D tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must be same for both A and B");
    TORCH_CHECK(A.size(1) == B.size(1), "For AB^T A and B's rows should match");
    TORCH_CHECK(A.size(2) == B.size(2), "For AB^T A and B's columns should match");

    int batch_size = A.size(0);
    int L = A.size(1);
    int D = A.size(2);

    auto C = torch::zeros({batch_size, L, L}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

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
    m.def("bmm", &batched_gemm_tiled_ABt_forward, "Batched Tiled GEMM AB^T CUDA forward pass");
}
