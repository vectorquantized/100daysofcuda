
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

__global__ void gemm_tiled(float* a, float* b, float* c, int M, int K, int N, float scale) {
    
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
        c[row * N + col] = p_value * scale;
    }
}

__global__ void transpose_tiled(const float* input, float* output, int rows, int cols) {
    
    __shared__ float copy_tile[TILE_WIDTH][TILE_WIDTH + 1];

    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;

    if (row < rows && col < cols) {
        copy_tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();

    int row_t = threadIdx.x + blockIdx.x * TILE_WIDTH;
    int col_t = threadIdx.y + blockIdx.y * TILE_WIDTH;

    if(row_t < cols && col_t < rows) {
        output[col_t * cols + row_t] = copy_tile[threadIdx.x][threadIdx.y]; 
    }
}

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

void self_attention(ten::Tensor Q, ten::Tensor K, ten::Tensor V, ten::Tensor output, int B, int L, int D) {
    /* Q: Query matrix shape: (B, L, D)
       K: Key matrix shape: (B, L, D)
       V: Value matrix shape: (B, L, D)
       attn_scores = softmax(QK^T/scale) * v
       output = W * attn_scores
    */
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks_per_grid((D + TILE_WIDTH - 1) / TILE_WIDTH, (L + TILE_WIDTH - 1) / TILE_WIDTH);

    ten::Tensor K_T = ten::zeros(B, D, L);
    for (int b = 0; b < B; ++b) {
        transpose_tiled<<<blocks_per_grid, threads_per_block>>>(K.get(b), K_T.get(b), L, D);
    }

    float scale = 1.0f / sqrtf(D);
    ten::Tensor scores = ten::zeros(B, L, L);
    for(int b = 0; b < B; b++) {
        gemm_tiled<<<blocks_per_grid, threads_per_block>>>(Q.get(b), K_T.get(b), scores.get(b), L, D, L, scale);
    }

    int num_threads = std::min(static_cast<int>(B * L), NUM_THREADS);
    dim3 block_dim(num_threads);
    dim3 grid_dim((B * L + num_threads - 1) / num_threads);
    
    batched_online_softmax<<<grid_dim, block_dim>>>(scores, scores, B, L, D); 
    for(int b = 0; b < B; ++b) {
        gemm_tiled<<<blocks_per_grid, threads_per_block>>>(scores.get(b), V.get(b), output.get(b), L, L, D, 1.0f);
    }
}

int main() {
    // Define dimensions
    const int B = 4;  // Batch size
    const int L = 32; // Sequence length
    const int D = 256; // Embedding dimension

    unsigned int baseSeed = 42;

    // Allocate pinned memory for host vectors
    PinnedVector<float> Q_h(B * L * D);
    PinnedVector<float> K_h(B * L * D);
    PinnedVector<float> V_h(B * L * D);
    PinnedVector<float> output_h(B * L * D);

    // Initialize input tensors with random values
    cpu_utils::init_random_vector(Q_h, B * L * D, baseSeed);
    cpu_utils::init_random_vector(K_h, B * L * D, baseSeed + 1);
    cpu_utils::init_random_vector(V_h, B * L * D, baseSeed + 2);

    // Declare device tensors
    ten::Tensor Q_d, K_d, V_d, output_d;

    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        Q_d.allocate(B * L * D, 3);
        K_d.allocate(B * L * D, 3);
        V_d.allocate(B * L * D, 3);
        output_d.allocate(B * L * D, 3);
    }

    // Copy data from Host to Device
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(Q_d.data, Q_h.data(), B * L * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(K_d.data, K_h.data(), B * L * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(V_d.data, V_h.data(), B * L * D * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Execute Self-Attention Kernel
    {
        TIMED_CUDA_BLOCK("🚀 Self-Attention Kernel Execution");
        self_attention(Q_d, K_d, V_d, output_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
    }

    // Copy results back to Host
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyDeviceToHost)");
        CUDA_ERROR_CHECK(cudaMemcpy(output_h.data(), output_d.data, B * L * D * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Free device memory
    Q_d.free();
    K_d.free();
    V_d.free();
    output_d.free();

    std::cout << "✅ Self-Attention completed successfully!\n";
    return 0;
}