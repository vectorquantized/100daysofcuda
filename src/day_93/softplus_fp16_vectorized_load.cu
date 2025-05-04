#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "csrc/timing_utils.h"
#include "cuda/tensor.h"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define NUM_THREADS 256


__global__ void softplus(const half* input, float* output, size_t B, size_t L, size_t D) {
    int batch_seq_idx = blockIdx.y;
    int batch_idx = batch_seq_idx / L;
    int seq_idx = batch_seq_idx % L;
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int vec_D = D / 2;
    int base_idx = batch_idx * L * D + seq_idx * D;

    if (vec_idx < vec_D) {
        const half2* input_vec = reinterpret_cast<const half2*>(input + base_idx);
        float2 x = __half22float2(input_vec[vec_idx]);
        float2 y;
        y.x = log1pf(expf(x.x));
        y.y = log1pf(expf(x.y));

        output[base_idx + 2 * vec_idx] = y.x;
        output[base_idx + 2 * vec_idx + 1] = y.y;
    }


    // Handle odd element if D is not multiple of 2
    if ((vec_idx == vec_D) && (D % 2 != 0)) {
        int scalar_idx = base_idx + D - 1;
        float x = __half2float(input[scalar_idx]);
        output[scalar_idx] = log1pf(expf(x));
    }
}


void kernel_launch(const ten::Tensor& a_d, const ten::Tensor& b_d, size_t B, size_t L, size_t D) {

    int vec_D = D / 2 + (D % 2);
    dim3 threads_per_block(NUM_THREADS);
    dim3 blocks_per_grid ((vec_D + NUM_THREADS - 1) / NUM_THREADS, B * L);
    softplus<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<const half*>(a_d.data), b_d.data, B, L, D);
}

void softplus_cpu(const float* input, float* output, size_t B, size_t L, size_t D) {
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                size_t idx = b * L * D + l * D + d;
                output[idx] = std::log1pf(expf(input[idx]));
            }
        }
    }
}




int main(int argc, char* argv[]) {
    int B = 4;
    size_t M = 2048;
    size_t N = 4096;

    unsigned int baseSeed = 42;
    PinnedVector<float> a_h(B * M * N);
    PinnedVector<__half> a_h_fp16(B * M * N);
    PinnedVector<float> b_h(B * M * N);
    PinnedVector<float> b_ref(B * M * N);
    cpu_utils::init_random_vector(a_h, B * M * N, baseSeed);
    for (size_t i = 0; i < B * M * N; ++i) {
        a_h_fp16[i] = __float2half(a_h[i]);
    }
    ten::Tensor a_d, b_d;
    
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        a_d.allocate(B * M * N, 3);
        b_d.allocate(B * M * N, 3);
    }
    
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h_fp16.data(), B * M * N * sizeof(__half), cudaMemcpyHostToDevice));
    }

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch(a_d, b_d, B, M, N);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        constexpr size_t col_width = 40;
        
        std::cout << std::left << std::setw(col_width) <<  "🚀 Kernel execution time"
                  << ":  " << std::fixed << std::setprecision(3) 
                  << std::setw(8) << milliseconds << " ms" << std::endl;
    }

    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyDeviceToHost)");
        CUDA_ERROR_CHECK(cudaMemcpy(b_h.data(), b_d.data, B * M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    a_d.free();
    b_d.free();
    

    {
        TIMED_CPU_FUNCTION();
        softplus_cpu(a_h.data(), b_ref.data(), B, M, N);
    }

    // Validation: compare b_h and b_ref
    bool all_close = true;
    float tol = 1e-3f;
    for (size_t i = 0; i < B * M * N; ++i) {
        float diff = std::abs(b_h[i] - b_ref[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << b_h[i] << " CPU=" << b_ref[i] << " (diff=" << diff << ")\n";
            all_close = false;
            break;
        }
    }

    if (all_close) {
        std::cout << "✅ Output matches CPU reference within tolerance.\n";
    } else {
        std::cerr << "❌ Output mismatch detected.\n";
    }

    return 0;
}