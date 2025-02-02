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


__global__ void softmax_tiled(ten::Tensor input, ten::Tensor output,
                            size_t M, size_t N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;

    __shared__ float shared_max[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float row_sums[TILE_WIDTH];

    if (row >= M) return;

    // Step 1: Find max value
    float thread_max = -FLT_MAX;
    for (int col = tx; col < N; col += blockDim.x) {
        if (col < N) {
            thread_max = fmaxf(thread_max, input[row * N + col]);
        }
    }
    shared_max[ty][tx] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_max[ty][tx] = fmaxf(shared_max[ty][tx], shared_max[ty][tx + stride]);
        }
        __syncthreads();
    }

    float max_val = shared_max[ty][0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = 0; i < (N + blockDim.x - 1) / blockDim.x; i++) {
        int col = i * blockDim.x + tx;
        if (col < N) {
            float val = expf(input[row * N + col] - max_val);
            thread_sum += val;
        }
    }

    // Store thread sums for reduction
    shared_max[ty][tx] = thread_sum;
    __syncthreads();

    if (tx == 0) {
        float row_sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            row_sum += shared_max[ty][i];
        }
        row_sums[ty] = row_sum;
    }
    __syncthreads();

    float final_sum = row_sums[ty];
     
     for (int c = tx; c < N; c += TILE_WIDTH) {
        float value = expf(input[row * N + c] - max_val); // Stabilized exp
        output[row * N + c] = value / (final_sum + EPSILON);
    }
}
void kernel_launch(const ten::Tensor& input, const ten::Tensor& output, size_t M, size_t N) {

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks_per_grid (
            (N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH
        );
    
    softmax_tiled<<<blocks_per_grid, threads_per_block>>>(input, output, M, N);
    
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
    a_d.allocate(M * N);
    b_d.allocate(M * N);
    
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
    std::vector<float> b_ref = cpu_kernels::softmax<float>(a_h, M, N, TILE_WIDTH, EPSILON);
    COMPARE_RESULT(b_ref.data(), b_h.data(), M * N, 1e-5);
    return 0;
}