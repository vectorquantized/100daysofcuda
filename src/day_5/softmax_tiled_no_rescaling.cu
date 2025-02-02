#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define NUM_THREADS 256
#define TILE_WIDTH 16
#define EPSILON 1e-7

// Let's start by softmax basic kernel and modify it to make it into tiled version.
/*
For tiled version, we need to identify the HBM reads and writes, they are the bottleneck.
Copy over the tile to shared memory, make one thread operate on a tile.
*/

__global__ void softmax_tiled(ten::Tensor input, ten::Tensor output, size_t M, size_t N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;

    __shared__ float row_max[TILE_WIDTH];  
    __shared__ float row_sum[TILE_WIDTH];  

    if (row >= M) return; // Boundary check

    float max_value = -FLT_MAX;
    for (int c = tx; c < N; c += TILE_WIDTH) {
        max_value = fmaxf(max_value, input[row * N + c]);
    }

    row_max[ty] = max_value;
    __syncthreads();

    // Reduce max value across the row 
    for (int stride = TILE_WIDTH / 2; stride > 0; stride /= 2) {
        if (ty < stride) {
            row_max[ty] = fmaxf(row_max[ty], row_max[ty + stride]);
        }
        __syncthreads();
    }

    max_value = row_max[0];  // The final max value is stored in row_max[0]
    __syncthreads();

    float sum_exp = 0.0f;
    for (int c = tx; c < N; c += TILE_WIDTH) {
        float value = expf(input[row * N + c] - max_value); // Stabilized exp
        sum_exp += value;
    }

    __syncthreads();

    // Reduce sum across the row
    row_sum[ty] = sum_exp;
    __syncthreads();

    for (int stride = TILE_WIDTH / 2; stride > 0; stride /= 2) {
        if (ty < stride) {
            row_sum[ty] += row_sum[ty + stride];
        }
        __syncthreads();
    }

    sum_exp = row_sum[0];
    __syncthreads();


    for (int c = tx; c < N; c += TILE_WIDTH) {
        float value = expf(input[row * N + c] - max_value); // Stabilized exp
        output[row * N + c] = value / (sum_exp + EPSILON);
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
    std::vector<float> a_h(M * N);
    std::vector<float> b_h(M * N);
    cpu_utils::init_random_vector(a_h, M * N, baseSeed);
    
    ten::Tensor a_d, b_d;
    a_d.allocate(M * N);
    b_d.allocate(M * N);
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
    
    std::vector<float> b_ref = cpu_kernels::softmax_exact<float>(a_h, M, N, EPSILON);
    
    COMPARE_RESULT(b_ref.data(), b_h.data(), M * N, 1e-5);
    return 0;
}