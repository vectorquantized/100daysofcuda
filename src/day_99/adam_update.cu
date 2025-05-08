#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "../cuda/cuda_utils.h"
#include "../csrc/utils.h"

__global__ void adam_update(
    float* param,
    float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int step,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g = grad[i];
    float m_i = m[i];
    float v_i = v[i];

    m_i = beta1 * m_i + (1.0f - beta1) * g;
    v_i = beta2 * v_i + (1.0f - beta2) * g * g;

    float m_hat = m_i / (1.0f - powf(beta1, step));
    float v_hat = v_i / (1.0f - powf(beta2, step));

    param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

    m[i] = m_i;
    v[i] = v_i;
}

int main() {
    int N = 1024 * 1024;
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int step = 1;

    std::vector<float> param_h, grad_h, m_h, v_h;

    cpu_utils::init_random_vector(param_h, N, 42);
    cpu_utils::init_random_vector(grad_h, N, 43);
    cpu_utils::init_random_vector(m_h, N, 44);
    cpu_utils::init_random_vector(v_h, N, 45);

 

    float *param_d, *grad_d, *m_d, *v_d;
    {
        TIMED_CUDA_BLOCK("💾 Memory Allocation on Device");
        
        CUDA_ERROR_CHECK(cudaMalloc(&param_d, N * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&grad_d, N * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&m_d, N * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&v_d, N * sizeof(float)));
    }
    {
        TIMED_CUDA_BLOCK("💾 Mem copy (cudaMemcpyHostToDevice)");
        CUDA_ERROR_CHECK(cudaMemcpy(param_d, param_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(grad_d, grad_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(m_d, m_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(v_d, v_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    {
        TIMED_CUDA_BLOCK("🚀 Kernel execution time");
        adam_update<<<blocks, threads>>>(param_d, grad_d, m_d, v_d, lr, beta1, beta2, epsilon, step, N);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
    CUDA_ERROR_CHECK(cudaMemcpy(param_h.data(), param_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Sanity check on updated params (first 10): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << param_h[i] << " ";
    }
    std::cout << std::endl;

    

    cudaFree(param_d);
    cudaFree(grad_d);
    cudaFree(m_d);
    cudaFree(v_d);

    return 0;
}