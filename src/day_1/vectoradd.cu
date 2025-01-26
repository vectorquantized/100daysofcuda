#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"
#include "cuda/cuda_utils.h"

/*
The program's flow would be as follows:
1. Allocate memory on CPU and initialize that memory.
2. Allocate memory on GPU using cudaMalloc.
3. Copy data from Host to device using CudaMemcpy with the flag CudaMemcpyHostToDevice
4. Define kernel launch params
5. launch kernel, add
6. Copy data from device to host.
*/

// The definition of this structure is inspired by: https://github.com/1y33/100Days/blob/main/day04/layerNorm.cu#L6
struct Tensor {
    float* data;
    int size;
    int* shape;
    int dims;

    __host__ void allocate(int N) {
        size = N;
        CUDA_ERROR_CHECK(cudaMalloc((void **)&data, size * sizeof(float)));
    }

    //TODO: May be need shape access on device, allocate if needed.

    __host__ void free() {
        cudaFree(data);
    }
    __device__ float& operator[] (int i) {
        // add error checking, if needed.
        return data[i];
    }
};


__global__ void add(Tensor a, Tensor b, Tensor c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < a.size) {
        c[tid] = a[tid] + b[tid];
    }
}

void kernel_launch(Tensor a_d, Tensor b_d, Tensor c_d, size_t size) {
    TIMED_CUDA_FUNCTION();
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    add<<<blocks_per_grid, threads_per_block>>>(a_d, b_d, c_d);

    cudaDeviceSynchronize(); // Barrier sync
}

int main(int argc, char* argv[]) {
    size_t size = 1024; // we'd like to add the vectors of this size.
    unsigned int baseSeed = 42;
    std::vector<float> a_h;
    std::vector<float> b_h;
    std::vector<float> c_h;
    c_h.resize(size);
    init_random_vector(a, size, baseSeed);
    init_random_vector(b, size, baseSeed + 1);
    Tensor a_d, b_d, c_d;
    a_d.allocate(size);
    b_d.allocate(size);
    c_d.allocate(size);
    CUDA_ERROR_CHECK(cudaMemcpy(a_d.data, a_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(b_d.data, b_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    kernel_launch();
    CUDA_ERROR_CHECK(cudaMemcpy(c_h.data(), c_d.data, size * sizeof(float), cudaMemcpyDeviceToHost));
    a_d.free();
    b_d.free();
    c_d.free();

    return 0;
}
