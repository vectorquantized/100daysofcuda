#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define NUM_THREADS 256
#define TILE_WIDTH 32
#define EPSILON 1e-7


__global__ void online_softmax(ten::Tensor input, ten::Tensor output,
                            size_t M, size_t N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;

    if (row >= M) return;

    // Step 1: Find max value
    float thread_max = -FLT_MAX;
    float norm = 0.0f;
    for (int col = 0; col < N; ++col) {
        if (col < N) {
            float curr_value = input[row * N + col];
            if (curr_value > thread_max) {
                thread_max = curr_value;
                norm *= expf(thread_max - curr_value);
            }
            norm += curr_value;
        }
    }
    
     for (int c = tx; c < N; c += TILE_WIDTH) {
        float value = expf(input[row * N + c] - thread_max); // Stabilized exp
        output[row * N + c] = value / (norm + EPSILON);
    }
}
void kernel_launch(const ten::Tensor& input, const ten::Tensor& output, size_t M, size_t N) {

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks_per_grid (
            (N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH
        );
    online_softmax<<<blocks_per_grid, threads_per_block>>>(input, output, M, N);
    
}


int main(int argc, char* argv[]) {
    size_t M = 8192;
    size_t N = 4096;

    unsigned int baseSeed = 42;
    // Use pinned memory for std::vector
    PinnedVector<float> a_h(M * N);
    PinnedVector<float> b_h(M * N);

    cpu_utils::init_random_vector(a_h, M * N, baseSeed);
    
    ten::Tensor a_d, b_d;
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        a_d.allocate(M * N);
        b_d.allocate(M * N);
    }
    
    // Memcpy HostToDevice
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    {
        TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        kernel_launch(a_d, b_d, M, N);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyDeviceToHost)");
        CUDA_ERROR_CHECK(cudaMemcpy(b_h.data(), b_d.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    a_d.free();
    b_d.free();
    
    // Convert to standard vector
    std::vector<float> b_ref(a_h.size());
    cpu_kernels::online_softmax<float>(a_h, b_ref, M, N, EPSILON);
    COMPARE_RESULT(b_ref.data(), b_h.data(), M * N, 1e-5);
    return 0;
}