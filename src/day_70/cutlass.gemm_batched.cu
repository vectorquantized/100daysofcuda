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
    float beta = 0.0;
    int lda = M;
    int ldb = K;
    int ldc = M;
    int batch_stride_A = lda * K;
    int batch_stride_B = ldb * N;
    int batch_stride_C = ldc * N;

    int count_A = batch_count * lda * K;
    int count_B = batch_count * ldb * N;
    int count_C = batch_count * ldc * N;

    std::vector<float> host_A(count_A);
    std::vector<float> host_B(count_B);
    std::vector<float> host_C(count_C);

    auto init_A = [kRange, lda, K] (int batch, int row, int col) -> float {
        return static_cast<float>((batch * lda * K + col * lda + row) % kRange);
    };
    
    auto init_B = [kRange, N, K, ldb, batch_count](int batch, int row, int col) -> float {
        return static_cast<float>(((N + K * ldb + batch_count * ldb * K)-(batch * ldb * K + col * ldb + row)) % kRange);
    };
    
    auto init_C = [](int batch, int row, int col) -> float {
        return 1.0f;
    };
    
    cpu_utils::initialize_batched_matrices_col_major(host_A.data(), batch_count, M, K, lda, lda * K, init_A);
    cpu_utils::initialize_batched_matrices_col_major(host_B.data(), batch_count, K, N, ldb, ldb * N, init_B);
    cpu_utils::initialize_batched_matrices_col_major(host_C.data(), batch_count, M, N, ldc, ldc * N, init_C);

    float* A;
    CUDA_ERROR_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
    float* B;
    CUDA_ERROR_CHECK(cudaMalloc(&B, count_B * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
    float* C;
    CUDA_ERROR_CHECK(cudaMalloc(&C, count_C * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));

    cutlass::Status status;
    if(deviceProp.major == 8) {
        using AmpereConfig = GemmConfig<cutlass::arch::Sm80>;
        status = run_gemm_batched<float>(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, batch_count);
    } else if (deviceProp.major ==7 && deviceProp.minor == 5) {
        using TuringConfig = GemmConfig<cutlass::arch::Sm75>;
        status = run_gemm_batched<float>(M, N, K, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C, beta, batch_count);
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

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}