#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"

#define TILE_WIDTH 16
#define COARSENING_FACTOR 2

torch::Tensor batched_gemm_tiled_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(B.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(A.ndimension() == 3 && B.ndimension() == 3 , "A and B must be 3D tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must be same for both A and B");
    TORCH_CHECK(A.size(2) == B.size(1), "A's columns should match B's rows for matmul.");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );
    
    batched_gemm_tiled<float, TILE_WIDTH><<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, M, K, N, scale);
    // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return C;
}

torch::Tensor batched_gemm_tiled_coarsened_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(B.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(A.ndimension() == 3 && B.ndimension() == 3 , "A and B must be 3D tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must be same for both A and B");
    TORCH_CHECK(A.size(2) == B.size(1), "A's columns should match B's rows for matmul.");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size
    );
    
    bgemm_tiled_coarsened<float, TILE_WIDTH, COARSENING_FACTOR><<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, M, K, N, scale);
    // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &batched_gemm_tiled_forward, "Batched Tiled GEMM CUDA forward pass");
    m.def("bmm_coarsened", &batched_gemm_tiled_coarsened_forward, "Batched Tiled GEMM CUDA forward pass");
}
