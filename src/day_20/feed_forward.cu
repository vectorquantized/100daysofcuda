#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/cublas_gemm.h"
#include "../cuda/cuda_utils.h"
#include "../cuda/activations.h"

#define TILE_WIDTH 16
#define CEIL_DIV(M, N) ((M + N - 1 ) / N)

torch::Tensor swiglu_forward(torch::Tensor up, torch::Tensor gate, torch::Tensor down, torch::Tensor A) {
    TORCH_CHECK(up.device().is_cuda() , "up projection matrix must be a cuda tensor");
    TORCH_CHECK(gate.device().is_cuda() , "gate projection matrix must be a cuda tensor");
    TORCH_CHECK(down.device().is_cuda() , "down projection matrix must be a cuda tensor");
    TORCH_CHECK(A.size(2) == up.size(1), "input last dim should match up projection's input dim");
    TORCH_CHECK(up.size(0) == gate.size(0) && up.size(1) == gate.size(1), "up projection should match with gate projection");
    TORCH_CHECK(up.size(0) == down.size(1), "up projection's out dim should match with down projection's input dim.");
    TORCH_CHECK(down.size(0) == A.size(2), "down projection's out dim should match with input's last dim.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(up.dtype() == torch::kFloat32, "up must be float32");
    TORCH_CHECK(gate.dtype() == torch::kFloat32, "gate must be float32");
    TORCH_CHECK(down.dtype() == torch::kFloat32, "down must be float32");

    int B = A.size(0);
    int L = A.size(1);
    int D = A.size(2);
    int H = up.size(0);

    std::cout << "up size: " << up.sizes() << std::endl;
    std::cout << "gate size: " << gate.sizes() << std::endl;
    std::cout << "down size: " << down.sizes() << std::endl;
    auto up_proj = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    cublas_lt_matmul<float>(A.data_ptr<float>(), up.data_ptr<float>(), up_proj.data_ptr<float>(), 1, B * L, D, H);

    auto gate_proj = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    cublas_lt_matmul<float>(A.data_ptr<float>(), gate.data_ptr<float>(), gate_proj.data_ptr<float>(), 1, B * L, D, H);


    int num_elements = B * L * H;
    int block_size = TILE_WIDTH * TILE_WIDTH;
    dim3 block_dim(block_size);
    dim3 grid_dim(CEIL_DIV(num_elements, block_size));
    auto output = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    swiglu<float><<<grid_dim, block_dim>>>(up_proj.data_ptr<float>(), 
                                           gate_proj.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           B, L, H);
    
    auto down_proj = torch::zeros({B, L, D}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    cublas_lt_matmul<float>(output.data_ptr<float>(), down.data_ptr<float>(), down_proj.data_ptr<float>(), 1, B*L, H, D);
    return down_proj;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swiglu_forward, "Swiglu CUDA forward pass");
}
