#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../cuda/cuda_utils.h"

#define NUM_THREADS 256

__global__ void rmsnorm_dgrad(
    const float* dL_dy,  // [B, L, D]
    const float* x,      // [B, L, D]
    const float* gamma,  // [D]
    const float* rms,    // [B, L]
    float* dL_dx,        // [B, L, D]
    int B, int L, int D
) {
    int b = blockIdx.z;
    int l = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= D) return;

    __shared__ float shared_dot[256];
    float local_dot = 0.0f;

    if (tid < D) {
        int idx = ((b * L + l) * D) + tid;
        local_dot = dL_dy[idx] * gamma[tid] * x[idx];
    }

    float dot = local_dot;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_dot[threadIdx.x / warpSize] = dot;
    }
    __syncthreads();

    if (threadIdx.x < 32) {  // reduce warps
        float warp_sum = (threadIdx.x < D/warpSize) ? shared_dot[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (threadIdx.x == 0) shared_dot[0] = warp_sum;
    }
    __syncthreads();

    dot = shared_dot[0];

    float rms_val = rms[b * L + l];
    float denom = D * rms_val * rms_val;

    if (tid < D) {
        int idx = ((b * L + l) * D) + tid;
        float scale = x[idx] * dot / denom;
        dL_dx[idx] = dL_dy[idx] * gamma[tid] / rms_val - scale;
    }
}

int main() {
    int B = 16;
    int L = 2048;
    int D = 8192;

    size_t size = B * L * D;
    size_t size_rms = B * L;

    std::vector<float> x_h(size);
    std::vector<float> dL_dy_h(size);
    std::vector<float> gamma_h(D);
    std::vector<float> rms_h(size_rms);
    std::vector<float> dL_dx_h(size, 0);

    for (int i = 0; i < size; ++i) {
        x_h[i] = (i % 5) * 0.1f + 0.5f;
        dL_dy_h[i] = (i % 3) * 0.2f + 0.3f;
    }
    for (int i = 0; i < D; ++i) {
        gamma_h[i] = 1.0f + 0.1f * i;
    }
    for (int i = 0; i < size_rms; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < D; ++j) {
            sum += x_h[i * D + j] * x_h[i * D + j];
        }
        rms_h[i] = std::sqrt(sum / D + 1e-6f);
    }

    float *x_d, *dL_dy_d, *gamma_d, *rms_d, *dL_dx_d;
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        CUDA_ERROR_CHECK(cudaMalloc(&x_d, size * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&dL_dy_d, size * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&gamma_d, D * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&rms_d, size_rms * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&dL_dx_d, size * sizeof(float)));
    }
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(x_d, x_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(dL_dy_d, dL_dy_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(gamma_d, gamma_h.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(rms_d, rms_h.data(), size_rms * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(NUM_THREADS);
    dim3 grid((D + block.x - 1) / block.x, L, B);

    {
        TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        rmsnorm_dgrad<<<grid, block>>>(dL_dy_d, x_d, gamma_d, rms_d, dL_dx_d, B, L, D);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
    

    CUDA_ERROR_CHECK(cudaMemcpy(dL_dx_h.data(), dL_dx_d, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(x_d);
    cudaFree(dL_dy_d);
    cudaFree(gamma_d);
    cudaFree(rms_d);
    cudaFree(dL_dx_d);

    return 0;
}