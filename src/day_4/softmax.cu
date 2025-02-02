#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define NUM_THREADS 256


__global__ void softmax(ten::Tensor input, ten::Tensor output, size_t M, size_t N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M) {
        float max_value = 0.0f;
        float row_sum = 0.0f;

        for (int col = 0; col < N; ++col) {
            max_value = max(max_value, input[row * N + col]);
        }

        for (int col = 0; col < N; ++col) {
            float value = expf(input[row * N + col] - max_value);
            row_sum += value;
        }

        for (int col = 0; col < N; ++col) {
            output[row * N + col] = expf(input[row * N + col] - max_value) / row_sum;
        }
    }
}


void kernel_launch(const ten::Tensor& a_d, const ten::Tensor& b_d, size_t M, size_t N) {
    int block_size_x = NUM_THREADS;
    int block_size_y = NUM_THREADS;

    dim3 threads_per_block(block_size_x, block_size_y);
    dim3 blocks_per_grid ((M + block_size_y - 1) / block_size_y);
    softmax<<<blocks_per_grid, threads_per_block>>>(a_d, b_d, M, N);
}




int main(int argc, char* argv[]) {
    size_t M = 8192;
    size_t N = 4096;

    unsigned int baseSeed = 42;
    PinnedVector<float> a_h(M * N);
    PinnedVector<float> b_h(M * N);
    cpu_utils::init_random_vector(a_h, M * N, baseSeed);
    
    ten::Tensor a_d, b_d;
    a_d.allocate(M * N);
    b_d.allocate(M * N);
    
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        a_d.allocate(M * N);
        b_d.allocate(M * N);
    }
    
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch(a_d, b_d, M, N);
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
        CUDA_ERROR_CHECK(cudaMemcpy(b_h.data(), b_d.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    a_d.free();
    b_d.free();
    
    std::vector<float> b_ref = cpu_kernels::softmax<float>(a_h);
    COMPARE_RESULT(b_ref.data(), b_h.data(), M * N, 1e-6);
    cpu_utils::print_vectors(b_ref.data(), b_h.data(), M*N);
    return 0;
}