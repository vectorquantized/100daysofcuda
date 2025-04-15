#include <cuda_runtime.h>
#include <math.h>
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


    int L = 1024;
    int D = 256;
    int batch_count = 8;
    int kRange = 17;

    // We're performing Q * K.T. We'll strore K in ColumMajor format, and while performing GEMM, we'll specify the layout as RowMajor
    // and by doing this we won't have to physically transpose to perform the dor product.
    // Shape of A/Q: (B, L, D)
    // Shape of B/K: (B, L, D)
    float alpha = 1.0/sqrtf(D);
    float beta = 0.0;
    int lda = L;
    int ldb = L; // We'll store K in ColumnMajor, and in GEMM send the Layout as RowMajor.
    int ldc = L;
    int ldv = L;
    int batch_stride_A = lda * D;
    int batch_stride_B = ldb * D;
    int batch_stride_C = ldc * L;
    int batch_stride_V = ldv * D;

    int count_A = batch_count * lda * D;
    int count_B = batch_count * ldb * D;
    int count_C = batch_count * ldc * L;
    int count_V = batch_count * ldv * D;

    std::vector<float> host_A(count_A);
    std::vector<float> host_B(count_B);
    std::vector<float> host_C(count_C);
    std::vector<float> host_V(count_V);

    auto init_A = [kRange, lda, D] (int batch, int row, int col) -> float {
        return static_cast<float>((batch * lda * D + col * lda + row) % kRange);
    };
    
    auto init_B = [kRange, L, D, ldb, batch_count](int batch, int row, int col) -> float {
        return static_cast<float>((batch * ldb * D + col * ldb + row) % kRange);
    };
    
    auto init_C = [](int batch, int row, int col) -> float {
        return 1.0f;
    };

    auto init_V = [kRange, L, D, ldv, batch_count](int batch, int row, int col) -> float {
        return static_cast<float>((batch * ldv * D + col * ldv + row) % kRange);
    };
    
    cpu_utils::initialize_batched_matrices_col_major(host_A.data(), batch_count, L, D, lda, lda * D, init_A);
    cpu_utils::initialize_batched_matrices_col_major(host_B.data(), batch_count, L, D, ldb, ldb * D, init_B);
    cpu_utils::initialize_batched_matrices_col_major(host_C.data(), batch_count, L, L, ldc, ldc * L, init_C);
    cpu_utils::initialize_batched_matrices_col_major(host_V.data(), batch_count, L, D, ldv, ldv * D, init_V);

    float* A;
    CUDA_ERROR_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
    float* B;
    CUDA_ERROR_CHECK(cudaMalloc(&B, count_B * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
    float* C;
    CUDA_ERROR_CHECK(cudaMalloc(&C, count_C * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));
    float* V;
    CUDA_ERROR_CHECK(cudaMalloc(&V, count_V * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(V, host_V.data(), count_V * sizeof(float), cudaMemcpyHostToDevice));

    
    cutlass::Status status = attention<float>(L, L, D, alpha, 
        A, lda, batch_stride_A, 
        B, ldb, batch_stride_B, 
        C, ldc, batch_stride_C, 
        V, ldv, batch_stride_V,
        beta, batch_count);
    
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
    cudaFree(V);

    return 0;
}