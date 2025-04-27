#include <cuda_runtime.h>
#include "../cuda/cutlass/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../csrc/utils.h"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);  // device 0
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Compute capability: " 
              << deviceProp.major << "." << deviceProp.minor << std::endl;


    int M = 1024;
    int N = 512;
    int K = 256;
    int batch_count = 8;
    int kRange = 17;

    float alpha = 1.0f;
    float beta = 1.0f; // We need to accumulate each batch's GEMM.
    int lda = N;
    int ldb = M;
    int ldc = N;
    int batch_stride_A = lda * M;
    int batch_stride_B = ldb * K;
    int batch_stride_C = 0;

    int count_A = batch_count * lda * M;
    int count_B = batch_count * ldb * K;
    int count_C = ldc * K;

    std::vector<float> host_A(count_A);
    std::vector<float> host_B(count_B);
    std::vector<float> host_C(count_C);

    auto init_A = [kRange, lda, M] (int batch, int row, int col) -> float {
        return static_cast<float>((batch * lda * M + col * lda + row) % kRange);
    };
    
    auto init_B = [kRange, N, K, ldb, batch_count](int batch, int row, int col) -> float {
        return static_cast<float>(((N + K * ldb + batch_count * ldb * K)-(batch * ldb * K + col * ldb + row)) % kRange);
    };
    
    auto init_C = [](int batch, int row, int col) -> float {
        return 0.0f;
    };
    
    cpu_utils::initialize_batched_matrices_col_major(host_A.data(), batch_count, N, M, lda, lda * M, init_A);
    cpu_utils::initialize_batched_matrices_col_major(host_B.data(), batch_count, M, K, ldb, ldb * K, init_B);
    cpu_utils::initialize_batched_matrices_col_major(host_C.data(), 1, N, K, ldc, ldc * K, init_C);

    /*
    We are doing the backprop implementation, specifically wgrad implementation for linear layer.
    Weight Tensor, denoted by W, is of Shape: N, K
    Input Tensor, denoted by x, is of Shape: B, M, K
    Output Tensor, denoted by y is give by:
    y = xW.T, Shape: B, M, N
    wgrad = dloss/dy * dy/dW, Shape: (N, K)
    dloss/dy = grad_output, Shape: (B, M, N)
    dy/dW = x, Shape: (B, M, K)
    wgrad = (grad_output[..., None] * x[:, :, None, :]).sum(dim=[0, 1])

    But we need to ensure that W is replicated, batch_stride_W should be 0 then.
    */
    float* A; // dloss/dy, grad_output, Shape: (B, M, N)
    CUDA_ERROR_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
    float* B; // x, Shape: (B, M, K)
    CUDA_ERROR_CHECK(cudaMalloc(&B, count_B * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
    float* C; // wgrad, Shape: (N, K)
    CUDA_ERROR_CHECK(cudaMalloc(&C, count_C * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));

    // per batch GEMM: (N, M) * (M, K) = (N, K).
    cutlass::Status status;
    if(deviceProp.major == 8) {
        using AmpereConfig = GemmConfig<cutlass::arch::Sm80>;
        status = run_gemm_batched<float>(N, K, M, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, batch_count);
    } else if (deviceProp.major ==7 && deviceProp.minor == 5) {
        using TuringConfig = GemmConfig<cutlass::arch::Sm75>;
        status = run_gemm_batched<float>(N, K, M, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, batch_count);
    } else {
        std::cerr << "Unsupported compute capability: "
                  << deviceProp.major << "." << deviceProp.minor << std::endl;
        return -1;
    }
    
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    if(status != cutlass::Status::kSuccess) {
        std::cout << "GEMM ERROR: " << static_cast<int>(status) << std::endl;
        return -1;
    }

    // --- dgrad computation: dX = dY * W ---
    // Allocate and copy weight matrix W (shape N×K) from host_B
    float* W;
    CUDA_ERROR_CHECK(cudaMalloc(&W, N * K * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(W, host_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    // Allocate output buffer dX (shape B×M×K)
    float* dX;
    int count_dX = batch_count * M * K;
    CUDA_ERROR_CHECK(cudaMalloc(&dX, count_dX * sizeof(float)));
    // Run batched GEMM: (M×N) * (N×K) = (M×K)
    status = run_gemm_batched<float>(
        M, K, N,
        1.0f,
        A, lda, batch_stride_A,   // A: dY (B×M×N)
        W, K, 0,                 // B: W (N×K), replicated
        dX, K, M * K,            // C: dX (B×M×K)
        0.0f,
        batch_count
    );
    cudaError_t cudaStatus2 = cudaDeviceSynchronize();
    if (cudaStatus2 != cudaSuccess) {
        std::cerr << "CUDA error in dgrad: " << cudaGetErrorString(cudaStatus2) << std::endl;
    }
    if(status != cutlass::Status::kSuccess) {
        std::cout << "dgrad GEMM ERROR: " << static_cast<int>(status) << std::endl;
        return -1;
    }
    cudaFree(W);
    cudaFree(dX);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}