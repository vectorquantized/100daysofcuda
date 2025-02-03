#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include "csrc/utils.h"
#include "csrc/timing_utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define NUM_THREADS 256
#define TILE_WIDTH 32
#define EPSILON 1e-7

// Let's get real now. We have a tensor of shape (B, L, D) and we want to apply softmax on 3rd (the hidden) dimension.
// We can accomplish this using a 1D grid first, benchmark it and then improve it.
__global__ void batched_online_softmax(ten::Tensor input, ten::Tensor output, size_t B, size_t L, size_t D) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= B * L) return;  // Add explicit bounds check

    int batch_idx = row / L;
    int seq_idx = row % L;
    
    if (batch_idx < B) {

        float thread_max = -FLT_MAX;
        float norm = 0.0f;
        int base_idx = batch_idx * L * D + seq_idx * D;
        for (int elem_idx = 0; elem_idx < D; ++elem_idx) {
            float curr_value = input[base_idx + elem_idx];
            if (curr_value > thread_max) {
                norm *= expf(thread_max - curr_value);
                thread_max = curr_value;
            }
            norm += expf(curr_value - thread_max);

        }

        for (int elem_idx = 0; elem_idx < D; ++elem_idx) {
            output[base_idx + elem_idx] = expf(input[base_idx + elem_idx] - thread_max) / (norm + EPSILON);
        }
    }
}

__global__ void batched_softmax(ten::Tensor input, ten::Tensor output, size_t B, size_t L, size_t D) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= B * L) return;  // Add explicit bounds check

    int batch_idx = row / L;
    int seq_idx = row % L;
    
    if (batch_idx < B) {
    
        float max_value = 0.0f;
        float row_sum = 0.0f;
        int base_idx = batch_idx * L * D + seq_idx * D;
        for (int elem_idx = 0; elem_idx < D; ++elem_idx) {
            max_value = max(max_value, input[base_idx + elem_idx]);
        }

        for (int elem_idx = 0; elem_idx < D; ++elem_idx) {
            float value = expf(input[base_idx + elem_idx] - max_value);
            row_sum += value;
        }

        for (int elem_idx = 0; elem_idx < D; ++elem_idx) {
            output[base_idx + elem_idx] = expf(input[base_idx + elem_idx] - max_value) / row_sum;
        }
    }
}

void kernel_launch(const ten::Tensor& input, const ten::Tensor& output, size_t B, size_t L, size_t D) {

    int num_threads = std::min(static_cast<int>(B * L), NUM_THREADS);
    dim3 threads_per_block(num_threads);
    dim3 blocks_per_grid((B * L + num_threads - 1) / num_threads);
    
    batched_online_softmax<<<blocks_per_grid, threads_per_block>>>(input, output, B, L, D);
}

void kernel_launch_batched(const ten::Tensor& input, const ten::Tensor& output, size_t B, size_t L, size_t D) {

    int num_threads = std::min(static_cast<int>(B * L), NUM_THREADS);
    dim3 threads_per_block(num_threads);
    dim3 blocks_per_grid((B * L + num_threads - 1) / num_threads);
    
    batched_softmax<<<blocks_per_grid, threads_per_block>>>(input, output, B, L, D);
}

void test_correctness() {
    std::cout << "Checking for Correctness" << std::endl;
    size_t B = 32;
    size_t L = 32;
    size_t D = 1024;

    unsigned int baseSeed = 42;
    // Use pinned memory for std::vector
    PinnedVector<float> a_h(B * L * D);
    PinnedVector<float> b_h(B * L * D);

    cpu_utils::init_random_vector(a_h, B * L * D, baseSeed);

    ten::Tensor a_d, b_d;
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        a_d.allocate(B * L * D);
        b_d.allocate(B * L * D);
    }
    
    // Memcpy HostToDevice
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), B * L * D * sizeof(float), cudaMemcpyHostToDevice));
    }

    {
        TIMED_CUDA_BLOCK("🚀 Online execution time");
        kernel_launch(a_d, b_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyDeviceToHost)");
        CUDA_ERROR_CHECK(cudaMemcpy(b_h.data(), b_d.data, B * L * D * sizeof(float), cudaMemcpyDeviceToHost));
    }

    a_d.free();
    b_d.free();

    std::vector<float> b_ref(a_h.size());
    b_ref.resize(a_h.size());
    {
        TIMED_CPU_FUNCTION();
        cpu_kernels::batched_softmax<float>(a_h, b_ref, B, L, D, EPSILON);
    }
     
    COMPARE_RESULT(b_ref.data(), b_h.data(), B * L * D, 1e-5);
    std::cout << "---------------------\n\n" << std::endl;
}

int main(int argc, char* argv[]) {

    test_correctness();
    std::cout << "Mesuring performance characteristics of online vs offline softmax" << std::endl;
    size_t B = 8;
    size_t L = 8192;
    size_t D = 16384;

    unsigned int baseSeed = 42;
    // Use pinned memory for std::vector
    PinnedVector<float> a_h(B * L * D);
    PinnedVector<float> b_h(B * L * D);

    cpu_utils::init_random_vector(a_h, B * L * D, baseSeed);
    
    ten::Tensor a_d, b_d;
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        a_d.allocate(B * L * D);
        b_d.allocate(B * L * D);
    }
    
    // Memcpy HostToDevice
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), B * L * D * sizeof(float), cudaMemcpyHostToDevice));
    }

    {
        TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch_batched(a_d, b_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("🚀 Online Kernel execution time");
        kernel_launch(a_d, b_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch_batched(a_d, b_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("🚀 Online Kernel execution time");
        kernel_launch(a_d, b_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyDeviceToHost)");
        CUDA_ERROR_CHECK(cudaMemcpy(b_h.data(), b_d.data, B * L * D * sizeof(float), cudaMemcpyDeviceToHost));
    }

    a_d.free();
    b_d.free();
    return 0;
}