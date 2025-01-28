#include <iostream>
#include <cuda_runtime.h>
#include "csrc/utils.h"
#include "cpu/kernels.h"
#include "cuda/cuda_utils.h"
#include "cuda/tensor.h"

/*
The program's flow would be as follows:
1. Allocate memory on CPU and initialize that memory.
2. Allocate memory on GPU using cudaMalloc.
3. Copy data from Host to device using CudaMemcpy with the flag CudaMemcpyHostToDevice
4. Define kernel launch params
5. launch kernel, add
6. Copy data from device to host.
7. Validate
*/


__global__ void add(ten::Tensor a, ten::Tensor b, ten::Tensor c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < a.size) {
        c[tid] = a[tid] + b[tid];
    }
}

void kernel_launch(ten::Tensor a_d, ten::Tensor b_d, ten::Tensor c_d, size_t size) {
    TIMED_CUDA_FUNCTION();
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    add<<<blocks_per_grid, threads_per_block>>>(a_d, b_d, c_d);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize()); // Barrier sync
}

int main(int argc, char* argv[]) {
    size_t size = 1024; // we'd like to add the vectors of this size.
    unsigned int baseSeed = 42;
    std::vector<float> a_h(size);
    std::vector<float> b_h(size);
    std::vector<float> c_h(size);
    cpu_utils::init_random_vector(a_h, size, baseSeed);
    cpu_utils::init_random_vector(b_h, size, baseSeed + 1);
    ten::Tensor a_d, b_d, c_d;
    a_d.allocate(size);
    b_d.allocate(size);
    c_d.allocate(size);
    CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(b_d.data, b_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    kernel_launch(a_d, b_d, c_d, size);
    CUDA_ERROR_CHECK(cudaMemcpy(c_h.data(), c_d.data, size * sizeof(float), cudaMemcpyDeviceToHost));
    a_d.free();
    b_d.free();
    c_d.free();

    std::vector<float> c_ref(size);
    cpu_kernels::vector_add(a_h, b_h, c_ref);
    COMPARE_RESULT(c_ref.data(), c_h.data(), size, 1e-4);
    return 0;
}
