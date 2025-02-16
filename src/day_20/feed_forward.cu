#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../cuda/activations.h"

#define TILE_WIDTH 16
#define CEIL_DIV(M, N) ((M + N - 1 ) / N)


torch::Tensor feed_forward(torch::Tensor up, torch::Tensor gate, torch::Tensor down, torch::Tensor A) {
    /**
        Shapes:
            input:  B, L, D
            up:     D, H => up_proj:    input @ up:                 B, L, H
            gate:   D, H => gate_proj:  input @ gate:               B, L, H
            down:   H, D => down_proj:  (up * silu(gate)) @ down:   B, L, D ((B, L, H) @ (H, D))
    */
    TORCH_CHECK(up.device().is_cuda() , "up projection matrix must be a cuda tensor");
    TORCH_CHECK(gate.device().is_cuda() , "gate projection matrix must be a cuda tensor");
    TORCH_CHECK(down.device().is_cuda() , "down projection matrix must be a cuda tensor");
    TORCH_CHECK(A.size(2) == up.size(0), "input last dim should match up projection's input dim");
    TORCH_CHECK(up.size(0) == gate.size(0) && up.size(1) == gate.size(1), "up projection should match with gate projection");
    TORCH_CHECK(up.size(1) == down.size(0), "up projection's out dim should match with down projection's input dim.");
    TORCH_CHECK(down.size(1) == A.size(2), "down projection's out dim should match with input's last dim.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(up.dtype() == torch::kFloat32, "up must be float32");
    TORCH_CHECK(gate.dtype() == torch::kFloat32, "gate must be float32");
    TORCH_CHECK(down.dtype() == torch::kFloat32, "down must be float32");

    int B = A.size(0);
    int L = A.size(1);
    int D = A.size(2);
    int H = up.size(1);

    auto up_proj = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    bmm_broadcast_B_launcher<float, TILE_WIDTH, float>(A.data_ptr<float>(), up.data_ptr<float>(), up_proj.data_ptr<float>(), B, L, D, H, 1.0f);
    // cublas_lt_matmul<float>(A.data_ptr<float>(), up.data_ptr<float>(), up_proj.data_ptr<float>(), B, L, D, H);

    auto gate_proj = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    bmm_broadcast_B_launcher<float, TILE_WIDTH, float>(A.data_ptr<float>(), gate.data_ptr<float>(), gate_proj.data_ptr<float>(), B, L, D, H, 1.0f);
    // cublas_lt_matmul<float>(A.data_ptr<float>(), gate.data_ptr<float>(), gate_proj.data_ptr<float>(), 1, B * L, D, H);


    int num_elements = B * L * H;
    int block_size = TILE_WIDTH * TILE_WIDTH;
    dim3 block_dim(block_size);
    dim3 grid_dim(CEIL_DIV(num_elements, block_size));
    auto output = torch::zeros({B, L, H}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    swiglu<float><<<grid_dim, block_dim>>>(up_proj.data_ptr<float>(), 
                                           gate_proj.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           B, L, H);
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));
    // cudaDeviceSynchronize();
    
    auto down_proj = torch::zeros({B, L, D}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    // cublas_lt_matmul<float>(output.data_ptr<float>(), down.data_ptr<float>(), down_proj.data_ptr<float>(), 1, B*L, H, D);
    bmm_broadcast_B_launcher<float, TILE_WIDTH, float>(output.data_ptr<float>(), down.data_ptr<float>(), down_proj.data_ptr<float>(), B, L, H, D, 1.0f);
    return down_proj;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &feed_forward, "FF CUDA forward pass");
}
