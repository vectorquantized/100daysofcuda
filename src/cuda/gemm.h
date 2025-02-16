#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>
#include "../cuda/cuda_utils.h"
#define TILE_WIDTH 16


template<typename T, int TileWidth, typename ScaleType = T>
__global__ void gemm_tiled(const T* A, const T* B, T* C, int M, int K, int N, ScaleType scale) {
    
    __shared__ T A_smem[TileWidth][TileWidth];
    __shared__ T B_smem[TileWidth][TileWidth];

    int row = threadIdx.y + blockIdx.y * TileWidth;
    int col = threadIdx.x + blockIdx.x * TileWidth;
    T p_value = static_cast<T>(0);

    for(int ph = 0; ph < (K + TileWidth - 1) / TileWidth; ++ph) {
        if(row < M && ph * TileWidth + threadIdx.x < K) {
            A_smem[threadIdx.y][threadIdx.x] = A[row * K + ph * TileWidth + threadIdx.x];
        } else {
            A_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }

        if(ph * TileWidth + threadIdx.y < K && col < N) {
            B_smem[threadIdx.y][threadIdx.x] = B[(ph * TileWidth + threadIdx.y) * N + col];
        } else {
            B_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        #pragma unroll
        for(int i =0; i < TileWidth; ++i) {
            p_value += A_smem[threadIdx.y][i] * B_smem[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = p_value * static_cast<T>(scale);
    }
}

template<typename T, int TileWidth, typename ScaleType>
__global__ void bmm_broadcast_B(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int batch_size, int M, int K, int N, ScaleType scale) {
    int batch_idx = blockIdx.z;
    int row = threadIdx.y + blockIdx.y * TileWidth;
    int col = threadIdx.x + blockIdx.x * TileWidth;

    __shared__ T A_smem[TileWidth][TileWidth];
    __shared__ T B_smem[TileWidth][TileWidth];

    const T* A_batch = A + batch_idx * M * K;
    // const T* B_batch = B + batch_idx * K * N;
    T* C_batch = C + batch_idx * M * N;

    T p_value = static_cast<T>(0);
    int num_tiles = (K + TileWidth - 1) / TileWidth;
    for(int ph = 0; ph < num_tiles; ++ph) {
        if (row < M && ph * TileWidth + threadIdx.x < K) {
            A_smem[threadIdx.y][threadIdx.x] = A_batch[row * K + ph * TileWidth + threadIdx.x];
        } else {
            A_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        if (ph * TileWidth + threadIdx.y < K && col < N) {
            B_smem[threadIdx.y][threadIdx.x] = B[(ph * TileWidth + threadIdx.y) * N + col];
        } else {
            B_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        for (int i = 0; i < TileWidth; ++i) {
            p_value += A_smem[threadIdx.y][i] * B_smem[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C_batch[row * N + col] = p_value * static_cast<T>(scale);
    }
}

template<typename T, int TileWidth, typename ScaleType>
__global__ void batched_gemm_tiled(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int batch_size, int M, int K, int N, ScaleType scale) {
    int batch_idx = blockIdx.z;
    int row = threadIdx.y + blockIdx.y * TileWidth;
    int col = threadIdx.x + blockIdx.x * TileWidth;

    __shared__ T A_smem[TileWidth][TileWidth];
    __shared__ T B_smem[TileWidth][TileWidth];

    const T* A_batch = A + batch_idx * M * K;
    const T* B_batch = B + batch_idx * K * N;
    T* C_batch = C + batch_idx * M * N;

    T p_value = static_cast<T>(0);
    int num_tiles = (K + TileWidth - 1) / TileWidth;
    for(int ph = 0; ph < num_tiles; ++ph) {
        if (row < M && ph * TileWidth + threadIdx.x < K) {
            A_smem[threadIdx.y][threadIdx.x] = A_batch[row * K + ph * TileWidth + threadIdx.x];
        } else {
            A_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        if (ph * TileWidth + threadIdx.y < K && col < N) {
            B_smem[threadIdx.y][threadIdx.x] = B_batch[(ph * TileWidth + threadIdx.y) * N + col];
        } else {
            B_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        for (int i = 0; i < TileWidth; ++i) {
            p_value += A_smem[threadIdx.y][i] * B_smem[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C_batch[row * N + col] = p_value * static_cast<T>(scale);
    }
}

template<typename T, int TileWidth, typename ScaleType>
__global__ void batched_gemm_tiled_ABT(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int batch_size, int L, int D, ScaleType scale) {
    int batch_idx = blockIdx.z;
    int row = threadIdx.y + blockIdx.y * TileWidth;
    int col = threadIdx.x + blockIdx.x * TileWidth;

    __shared__ T A_smem[TileWidth][TileWidth];
    __shared__ T B_smem[TileWidth][TileWidth];

    const T* A_batch = A + batch_idx * L * D;
    const T* B_batch = B + batch_idx * L * D;
    T* C_batch = C + batch_idx * L * L;

    T p_value = static_cast<T>(0);
    int num_tiles = (D + TileWidth - 1) / TileWidth;
    for(int ph = 0; ph < num_tiles; ++ph) {
        if (row < L && ph * TileWidth + threadIdx.x < D) {
            A_smem[threadIdx.y][threadIdx.x] = A_batch[row * D + ph * TileWidth + threadIdx.x];
        } else {
            A_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        if ((ph * TileWidth + threadIdx.y) < D && col < L) {
            B_smem[threadIdx.y][threadIdx.x] = B_batch[col * D + (ph * TileWidth + threadIdx.y)];
        } else {
            B_smem[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        for (int i = 0; i < TileWidth; ++i) {
            p_value += A_smem[threadIdx.y][i] * B_smem[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < L && col < L) {
        C_batch[row * L + col] = p_value * static_cast<T>(scale);
    }
}

template<typename T, int TileWidth, int CoarseningFactor, typename ScaleType>
__global__ void bgemm_tiled_coarsened(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int batch_size, int M, int K, int N, ScaleType scale) {
    int batch_idx = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = ty + blockIdx.y * TileWidth;
    int col_start = tx + blockIdx.x * TileWidth * CoarseningFactor;

    __shared__ T A_smem[TileWidth][TileWidth];
    __shared__ T B_smem[TileWidth][TileWidth];

    const T* A_batch = A + batch_idx * M * K;
    const T* B_batch = B + batch_idx * K * N;
    T* C_batch = C + batch_idx * M * N;

    int num_phases = CEIL_DIV(K, TileWidth);

    T p_value[CoarseningFactor] = {static_cast<T>(0)};
    for(int ph = 0; ph < num_phases; ++ph) {
        
        if (row < M && (ph * TileWidth + tx) < K) {
            A_smem[ty][tx] = A_batch[row * K + ph * TileWidth + tx];
        } else {
            A_smem[ty][tx] = static_cast<T>(0);
        }
        // now that we've loaded one tile of A in shared memory
        // let's use it to do more work per thread.
        for(int i = 0; i < CoarseningFactor; ++i) {
            int col = col_start + i * TileWidth;
            if (ph * TileWidth + ty < K && col < N) {
                B_smem[ty][tx] = B_batch[(ph * TileWidth + ty) * N + col];
            } else {
                B_smem[ty][tx] = static_cast<T>(0);
            }
            __syncthreads();
            for(int j = 0; j < TileWidth; ++j) {
                p_value[i] += A_smem[threadIdx.y][j] * B_smem[j][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for(int i = 0; i < CoarseningFactor; ++i) {
        int col = col_start + i * TileWidth;
        if (row < M && col < N) {
            C_batch[row * N + col] = p_value[i] * static_cast<T>(scale);
        }
    } 
}

template<typename T, int TileWidth, typename ScaleType>
void bmm_broadcast_B_launcher(const T* A, const T* B, T* C, int batch_size, int M, int K, int N, ScaleType scale) {
    /**
        * @brief Launches the batched GEMM tiled CUDA kernel with error checking.
        *
        * This function configures and launches the `batched_gemm_tiled` kernel for performing 
        * batched General Matrix-Matrix Multiplication (GEMM) using shared memory tiling.
        * It ensures error handling by checking for kernel launch and execution errors.
        *
        * @tparam T         Data type of matrix elements (e.g., float, double).
        * @tparam TileWidth Tile width used for shared memory tiling.
        * @tparam ScaleType Data type of the scaling factor applied to the matrix product.
        *
        * @param A          Pointer to the input matrix A of shape (batch_size, M, K).
        * @param B          Pointer to the input matrix B of shape (batch_size, K, N).
        * @param C          Pointer to the output matrix C of shape (batch_size, M, N).
        * @param batch_size Number of matrix multiplications to be performed in parallel.
        * @param M          Number of rows in matrices A and C.
        * @param K          Number of columns in matrix A and rows in matrix B.
        * @param N          Number of columns in matrices B and C.
        * @param scale      Scaling factor applied to the matrix multiplication result.
        *
        * @throws std::runtime_error if a CUDA kernel launch or execution error occurs.
    */
    dim3 block_dim(TileWidth, TileWidth);
    dim3 grid_dim(
        (N + TileWidth - 1) / TileWidth,
        (M + TileWidth - 1) / TileWidth,
        batch_size
    );
    bmm_broadcast_B<T, TileWidth, ScaleType><<<grid_dim, block_dim>>>(A, B, C, batch_size, M, K, N, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in file '%s' at line %i: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error in file '%s' at line %i: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#endif // GEMM_H