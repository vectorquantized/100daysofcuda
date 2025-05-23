#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "csrc/timing_utils.h"
#include "cuda/tensor.h"

#define NUM_THREADS 256


__global__ void softplus_dgrad(const float* grad_output, const float* input, float* grad_input, size_t B, size_t L, size_t D) {
    int batch_seq_idx = blockIdx.y;
    int batch_idx = batch_seq_idx / L;
    int seq_idx = batch_seq_idx % L;
    int feature_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int base_idx = batch_idx * L * D + seq_idx * D;

    if (feature_idx < D) {
        float x = input[base_idx + feature_idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float dgrad = grad_output[base_idx + feature_idx] * sigmoid_x;
        grad_input[base_idx + feature_idx] = dgrad;
    }
}


void kernel_launch(const ten::Tensor& grad_output_d, const ten::Tensor& a_d, const ten::Tensor& b_d, size_t B, size_t L, size_t D) {

    dim3 threads_per_block(NUM_THREADS);
    dim3 blocks_per_grid ((D + NUM_THREADS - 1) / NUM_THREADS, B * L);
    softplus_dgrad<<<blocks_per_grid, threads_per_block>>>(grad_output_d.data, a_d.data, b_d.data, B, L, D);
}

void softplus_backward_cpu(const float* grad_output, const float* input, float* grad_input, size_t B, size_t L, size_t D) {
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                size_t idx = b * L * D + l * D + d;
                float sigmoid_x = 1.0f / (1.0f + expf(-input[idx]));
                grad_input[idx] = grad_output[idx] * sigmoid_x;
            }
        }
    }
}




int main(int argc, char* argv[]) {
    int B = 4;
    size_t M = 2048;
    size_t N = 4096;

    unsigned int baseSeed = 42;
    PinnedVector<float> grad_output_h(B * M * N);
    PinnedVector<float> a_h(B * M * N);
    PinnedVector<float> b_h(B * M * N);
    PinnedVector<float> b_ref(B * M * N);
    cpu_utils::init_random_vector(a_h, B * M * N, baseSeed);
    cpu_utils::init_random_vector(grad_output_h, B * M * N, baseSeed + 1);
    
    ten::Tensor grad_output_d, a_d, b_d;
    
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        grad_output_d.allocate(B * M * N, 3);
        a_d.allocate(B * M * N, 3);
        b_d.allocate(B * M * N, 3);
    }
    
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), B * M * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(grad_output_d.data, grad_output_h.data(), B * M * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch(grad_output_d, a_d, b_d, B, M, N);
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
    grad_output_d.free();
    

    {
        TIMED_CPU_FUNCTION();
        softplus_backward_cpu(grad_output_h.data(), a_h.data(), b_ref.data(), B, M, N);
    }

    // Validation: compare b_h and b_ref
    bool all_close = true;
    float tol = 1e-4f;
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