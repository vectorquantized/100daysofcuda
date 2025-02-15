#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "../cuda/cublas_gemm.h"

torch::Tensor batched_gemm_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.ndimension() == 3 && B.ndimension() == 3, "A and B must be 3D tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match.");
    TORCH_CHECK(A.size(2) == B.size(1), "A's columns must match B's rows.");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    cublas_lt_matmul<float>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, M, K, N);
    return C;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_fast", &batched_gemm_forward, "Batched CUBLAS GEMM forward pass");
}