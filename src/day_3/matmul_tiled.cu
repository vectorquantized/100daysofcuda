#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

#define TILE_WIDTH 32

template <typename T, int TileWidth>
struct SharedMemory {
    T* data;

    __device__ SharedMemory(T* shared_ptr) : data(shared_ptr) {}
    
    __device__ void copyTile(const T* global_ptr, int row, int col, int stride, bool row_cond, bool col_cond) {
        if (row_cond && col_cond) {
            data[threadIdx.y * TileWidth + threadIdx.x] = global_ptr[row * stride + col];
        } else {
            data[threadIdx.y * TileWidth + threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();
    }
    
    __device__ T* operator[] (int y) { return &data[y * TileWidth]; }
    __device__ const T* operator[] (int y) const { return &data[y * TileWidth]; }
};

__global__ void gemm_tiled_with_struct(ten::Tensor a, ten::Tensor b, ten::Tensor c, int M, int K, int N) {
    __shared__ float shared_mem_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_mem_b[TILE_WIDTH][TILE_WIDTH];
    
    SharedMemory<float, TILE_WIDTH> a_shared(&shared_mem_a[0][0]);
    SharedMemory<float, TILE_WIDTH> b_shared(&shared_mem_b[0][0]);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * TILE_WIDTH;
    int col = tx + bx * TILE_WIDTH;

    float p_value = 0.0f;
    
    for(int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // Use SharedMemory struct's copyTile method
        a_shared.copyTile(a.data, row, ph * TILE_WIDTH + tx, K, row < M, ph * TILE_WIDTH + tx < K);
        b_shared.copyTile(b.data, ph * TILE_WIDTH + ty, col, N, ph * TILE_WIDTH + ty < K, col < N);

        #pragma unroll
        for(int k = 0; k < TILE_WIDTH; ++k) {
            if(ph * TILE_WIDTH + k < K) {
                p_value += a_shared[ty][k] * b_shared[k][tx];
            }
        }
        
        __syncthreads();
    }

    if(row < M && col < N) {
        c.data[row * N + col] = p_value;
    }
}


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

        #pragma unroll
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

// Define the correct function pointer type for your kernel
typedef void (*KernelFunction)(ten::Tensor, ten::Tensor, ten::Tensor, int, int, int);

void checkOccupancy(KernelFunction kernel, int blockSize) {
    int minGridSize, blockSizeReturned;
    CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeReturned, kernel, 0, 0));

    int maxBlocksPerSM;
    CUDA_ERROR_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, blockSize, 0));

    int device;
    cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device));

    // Prevent division by zero
    if (prop.maxThreadsPerMultiProcessor == 0 || blockSize == 0) {
        std::cerr << "Error: Invalid block size or max threads per SM" << std::endl;
        return;
    }

    float occupancy = static_cast<float>(maxBlocksPerSM) / (prop.maxThreadsPerMultiProcessor / blockSize);
    if (std::isnan(occupancy)) {
        std::cerr << "Error: NaN detected in occupancy calculation!" << std::endl;
        return;
    }


    // float occupancy = (float)maxBlocksPerSM / (prop.maxThreadsPerMultiProcessor / blockSize);
    std::cout << "Occupancy: " << occupancy * 100 << "%\n";
}

void kernel_launch(const ten::Tensor& a_d, const ten::Tensor& b_d, ten::Tensor& c_tiled, ten::Tensor& c_tiled_struct, size_t M, size_t K, size_t N) {
    TIMED_CUDA_FUNCTION();
    int block_size_x = TILE_WIDTH;
    int block_size_y = TILE_WIDTH;

    dim3 threads_per_block(block_size_x, block_size_y);
    dim3 blocks_per_grid (
                                (N + block_size_x - 1) / block_size_x, // output has N columns
                                (M + block_size_y - 1) / block_size_y // output has M rows
                            );
    cudaStream_t stream_tiled, stream_tiled_struct;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream_tiled));
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream_tiled_struct));

     // Check occupancy before launching the kernel
    checkOccupancy(gemm_tiled, TILE_WIDTH * TILE_WIDTH);
    checkOccupancy(gemm_tiled_with_struct, TILE_WIDTH * TILE_WIDTH);

    TIMED_CUDA_BLOCK("Stream tiled");
    gemm_tiled<<<blocks_per_grid, threads_per_block, 0, stream_tiled>>>(a_d, b_d, c_tiled, M, K, N);
    TIMED_CUDA_BLOCK("Stream tiled struct");
    gemm_tiled_with_struct<<<blocks_per_grid, threads_per_block, 0, stream_tiled_struct>>>(a_d, b_d, c_tiled_struct, M, K, N);

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_tiled)); // Barrier sync
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_tiled_struct)); 

    CUDA_ERROR_CHECK(cudaStreamDestroy(stream_tiled)); 
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream_tiled_struct)); 
}

int main(int argc, char* argv[]) {
    size_t M = 1024; 
    size_t K = 1024;
    size_t N = 2048;

    unsigned int baseSeed = 42;
    std::vector<float> a_h(M * K);
    std::vector<float> b_h(K * N);
    std::vector<float> c_h_tiled(M * N);
    std::vector<float> c_h_tiled_struct(M * N);
    cpu_utils::init_random_vector(a_h, M * K, baseSeed);
    cpu_utils::init_random_vector(b_h, K * N, baseSeed + 1);
    ten::Tensor a_d, b_d, c_tiled, c_tiled_struct;
    a_d.allocate(M * K);
    b_d.allocate(K * N);
    c_tiled.allocate(M * N);
    c_tiled_struct.allocate(M * N);
    CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(b_d.data, b_h.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    kernel_launch(a_d, b_d, c_tiled, c_tiled_struct, M, K, N);

    CUDA_ERROR_CHECK(cudaMemcpy(c_h_tiled.data(), c_tiled.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(c_h_tiled_struct.data(), c_tiled_struct.data, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    a_d.free();
    b_d.free();
    c_tiled.free();
    c_tiled_struct.free();
    std::vector<float> c_ref(M * N);
    cpu_kernels::gemm(a_h, b_h, c_ref, M, K, N);
    COMPARE_RESULT(c_ref.data(), c_h_tiled.data(), M*N, 1e-3);
    COMPARE_RESULT(c_ref.data(), c_h_tiled_struct.data(), M*N, 1e-3);
    return 0;
}