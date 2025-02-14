#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>
#define CEIL_DIV(M, N) ((M + N - 1 ) / N)

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

#endif // GEMM_H