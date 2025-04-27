#include <cuda_runtime.h>
#include "../cuda/cutlass/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../csrc/utils.h"
#include <iostream>
#include <vector>


int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);  // device 0
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Compute capability: " 
              << deviceProp.major << "." << deviceProp.minor << std::endl;


    int B = 32;   // batch size
    int N = 1;    // heads
    int L = 128;  // sequence length
    int D = 64;   // head dimension
    
    int ldq = L;
    int ldK = L;

    int count_Q = B * N * L * D;
    int count_K = B * 1 * L * D;
    int count_Scores = B * N * L * L;

    std::vector<float> host_Q(count_Q);
    std::vector<float> host_K(count_K);
    std::vector<float> host_Scores(count_Scores);

    int kRange = 100;

    auto init_Q = [kRange, N, ldq, D] (int batch, int head, int row, int col) -> float {
        return static_cast<float>((batch * N * ldq * D + head * ldq * D + col * ldq + row) % kRange);
    };
    
    auto init_K = [kRange, ldK, D](int batch, int head, int row, int col) -> float {
        return static_cast<float>((batch * ldK * D + col * ldK + row) % kRange);
    };
    
    auto init_Scores = [](int batch, int head, int row, int col) -> float {
        return 1.0f;
    };
    
    cpu_utils::initialize_batched_multi_headed_matrices_col_major(host_Q.data(), B * N, N, L, D, L, L * D, init_Q);
    cpu_utils::initialize_batched_multi_headed_matrices_col_major(host_K.data(), B, 1, L, D, L, L * D, init_K);
    cpu_utils::initialize_batched_multi_headed_matrices_col_major(host_Scores.data(), B * N, N, L, L, L, L * L, init_Scores);

    float* d_Q;
    CUDA_ERROR_CHECK(cudaMalloc(&d_Q, count_Q * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_Q, host_Q.data(), count_Q * sizeof(float), cudaMemcpyHostToDevice));
    float* d_K;
    CUDA_ERROR_CHECK(cudaMalloc(&d_K, count_K * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_K, host_K.data(), count_K * sizeof(float), cudaMemcpyHostToDevice));
    float* d_Scores;
    CUDA_ERROR_CHECK(cudaMalloc(&d_Scores, count_Scores * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_Scores, host_Scores.data(), count_Scores * sizeof(float), cudaMemcpyHostToDevice));

    cutlass::Status status;
    
    
    // We build host arrays of device pointers for array-batched MQA
    std::vector<float const*> ptrA(B * N);
    std::vector<float const*> ptrB(B * N);
    std::vector<float*> ptrC(B * N);

    for (int b = 0; b < B; ++b) {
        float const* baseQ = d_Q      + b * (N * L * D);
        float const* baseK = d_K      + b * (L * D);
        float*       baseC = d_Scores + b * (N * L * L);
        for (int n = 0; n < N; ++n) {
            int idx = b * N + n;
            ptrA[idx] = baseQ + n * (L * D);
            ptrB[idx] = baseK;  // reuse same K[b]
            ptrC[idx] = baseC + n * (L * L);
        }
    }

    status = run_multi_query_attention_gemm<float, 32, 1, 128, 64>(
        ptrA.data(), ptrB.data(), ptrC.data()
    );
    
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    if(status != cutlass::Status::kSuccess) {
        std::cout << "GEMM ERROR: " << static_cast<int>(status) << std::endl;
        return -1;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_Scores);

    return 0;
}