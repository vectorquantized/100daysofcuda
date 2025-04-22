#include <cuda_runtime.h>
#include "../cuda/cutlass/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../csrc/utils.h"
#include "../cuda/activations.h"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);  // device 0
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Compute capability: " 
              << deviceProp.major << "." << deviceProp.minor << std::endl;


    // A: batch_count, M, K => 8, 128, 2048
    // B: K, 4K => 256, 2048
    // C: K, 4K => 256, 2048
    // up = A @ B: batch_count, M, 4K
    // gate = A @ C: batch_count, M, 4K
    // swiglu(A): up * gate * sigmoid(gate) => (A @ B) * (A @ C) * sigmoid(A @ C)
    int M = 128;
    int K = 256;
    int N = 4 * K;
    int batch_count = 8;
    int kRange = 17;

    float alpha = 1.0f;
    float beta = 0.0;
    int lda = M;
    int ldup = K;
    int ldgate = K;
    int ldout = M;
    int batch_stride_A = lda * K;
    int batch_stride_up = ldout * N; 
    int batch_stride_gate = 0; // if I don't do this then I'd have to replicate up and gate weights.
    int batch_stride_out = ldout * N;

    int count_A = batch_count * lda * K;
    int count_up = 1 * ldup * N;
    int count_gate = 1 * ldgate * N;
    int count_y = batch_count * ldout * N;
    int count_out = batch_count * ldout * N;

    std::vector<float> host_A(count_A);
    std::vector<float> host_up(count_up);
    std::vector<float> host_gate(count_gate);
    std::vector<float> host_y(count_y);
    std::vector<float> host_out(count_out);

    auto init_A = [kRange, lda, K] (int batch, int row, int col) -> float {
        return static_cast<float>((batch * lda * K + col * lda + row) % kRange);
    };
    
    auto init_up = [kRange, N, K, ldup](int batch, int row, int col) -> float {
        return static_cast<float>((batch * ldup * K  + ldup * K + col * ldup + row) % kRange);
    };

    auto init_gate = [kRange, N, K, ldgate](int batch, int row, int col) -> float {
        return static_cast<float>((batch * ldgate * K + ldgate * K + col * ldgate + row) % kRange);
    };
    
    auto init_out = [](int batch, int row, int col) -> float {
        return 1.0f;
    };

    
    cpu_utils::initialize_batched_matrices_col_major(host_A.data(), batch_count, M, K, lda, lda * K, init_A);
    cpu_utils::initialize_batched_matrices_col_major(host_up.data(), 1, K, N, ldup, ldup * N, init_up);
    cpu_utils::initialize_batched_matrices_col_major(host_gate.data(), 1, K, N, ldgate, ldgate * N, init_gate);
    cpu_utils::initialize_batched_matrices_col_major(host_y.data(), batch_count, M, N, ldout, ldout * N, init_out);
    cpu_utils::initialize_batched_matrices_col_major(host_out.data(), batch_count, M, N, ldout, ldout * N, init_out);

    float* A;
    CUDA_ERROR_CHECK(cudaMalloc(&A, count_A * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
    float* up;
    CUDA_ERROR_CHECK(cudaMalloc(&up, count_up * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(up, host_up.data(), count_up * sizeof(float), cudaMemcpyHostToDevice));
    float* gate;
    CUDA_ERROR_CHECK(cudaMalloc(&gate, count_gate * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(gate, host_gate.data(), count_gate * sizeof(float), cudaMemcpyHostToDevice));
    float* y;
    CUDA_ERROR_CHECK(cudaMalloc(&y, count_y * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(y, host_y.data(), count_y * sizeof(float), cudaMemcpyHostToDevice));
    float* out;
    CUDA_ERROR_CHECK(cudaMalloc(&out, count_out * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(out, host_out.data(), count_out * sizeof(float), cudaMemcpyHostToDevice));

    cutlass::Status status_up;
    status_up = run_gemm_batched<float>(M, N, K, alpha, A, lda, batch_stride_A, up, ldup, batch_stride_up, y, ldout, batch_stride_out, beta, batch_count);

    using EpilogueOp = SwigluEpilogue<float, 128 / cutlass::sizeof_bits<float>::value, float, float>;

    typename EpilogueOp::Params params;
    params.alpha        = alpha;
    params.beta         = 1.0f;
    params.up_ptr       = y;
    params.ldm          = ldout;
    params.batch_stride = batch_stride_out;

    auto status_fused = run_fused_gemm_batched<float,EpilogueOp>(
        M, N, K,
        alpha,
        A, lda, batch_stride_A,
        gate, ldgate, batch_stride_gate,
        y, ldout, batch_stride_out,
        out, ldout, batch_stride_out,
        1.0f,
        batch_count,
        params
    );


    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    if(status_up != cutlass::Status::kSuccess) {
        std::cout << "GEMM ERROR in up projections: " << static_cast<int>(status_up) << std::endl;
        return -1;
    }

    if(status_fused != cutlass::Status::kSuccess) {
        std::cout << "GEMM ERROR in gate projection + fused swiglu: " << static_cast<int>(status_fused) << std::endl;
        return -1;
    }

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    cudaFree(A);
    cudaFree(up);
    cudaFree(gate);
    cudaFree(y);
    cudaFree(out);

    return 0;
}