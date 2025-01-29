#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define TILE_WIDTH 32

__global__ void gemm_tiled(ten::Tensor a, ten::Tensor b, ten::Tensor c, int M, int K, int N) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float a_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_shared[TILE_WIDTH][TILE_WIDTH];

    int row = ty  + by * TILE_WIDTH;
    int col = tx + bx * TILE_WIDTH;

    float p_value = 0.0f;
    int num_phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    // start phase wise computation.
    for(int ph = 0; ph< num_phases; ++ph) {
        // copy over a to a_shared memory space
        if(row < M && ph * TILE_WIDTH + tx < K) {
            a_shared[ty][tx] = a[row * K + ph * TILE_WIDTH + tx];
        } else {
            a_shared[ty][tx] = 0.0f;
        }

        // copy over b to b_shared memory space
        if (col < N && ph * TILE_WIDTH + ty < K) {
            b_shared[ty][tx] = b[(ph * TILE_WIDTH + ty) * N + col];
        } else {
            b_shared[ty][tx] = 0.0f;
        }
        // barrier sync, wait for all the threads to have the data.
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
                p_value += a_shared[ty][i] * b_shared[i][tx];
        }
        // barrier sync
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = p_value;
    }
}

void kernel_launch(const ten::Tensor& a_d, const ten::Tensor& b_d, ten::Tensor& c_d, size_t M, size_t K, size_t N) {
    TIMED_CUDA_FUNCTION();
    int block_size_x = TILE_WIDTH;
    int block_size_y = TILE_WIDTH;

    dim3 threads_per_block(block_size_x, block_size_y);
    dim3 blocks_per_grid (
                                (N + block_size_x - 1) / block_size_x, // output has N columns
                                (M + block_size_y - 1) / block_size_y // output has M rows
                            );
    gemm_tiled<<<blocks_per_grid, threads_per_block>>>(a_d, b_d, c_d, M, K, N);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
}

int main(int argc, char* argv[]) {
    size_t M = 1024; 
    size_t K = 1024;
    size_t N = 2048;

    unsigned int baseSeed = 42;
    std::vector<float> a_h(M * K);
    std::vector<float> b_h(K * N);
    std::vector<float> c_h(M * N);
    cpu_utils::init_random_vector(a_h, M * K, baseSeed);
    cpu_utils::init_random_vector(b_h, K * N, baseSeed + 1);
    ten::Tensor a_d, b_d, c_d;
    a_d.allocate(M * K);
    b_d.allocate(K * N);
    c_d.allocate(M * N);
    CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(b_d.data, b_h.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    kernel_launch(a_d, b_d, c_d, M, K, N);
    CUDA_ERROR_CHECK(cudaMemcpy(c_h.data(), c_d.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    a_d.free();
    b_d.free();
    c_d.free();
    std::vector<float> c_ref(M * N);
    cpu_kernels::gemm(a_h, b_h, c_ref, M, K, N);
    COMPARE_RESULT(c_ref.data(), c_h.data(), M*N, 1e-3);
    return 0;
}