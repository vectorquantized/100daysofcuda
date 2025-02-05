
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

__global__ void permute_kernel(ten::Tensor tensor, const int* perm) {
    if (threadIdx.x == 0) { 
        tensor.permute(perm);
    }
}

int main() {
    ten::Tensor tensor;
    int dims = 4;
    int shape[] = {2, 4, 3, 2}; // 3D tensor

    int total_size = 1;
    for (int i = 0; i < dims; i++) {
        total_size *= shape[i];
    }

    tensor.allocate(total_size, dims); // Allocate memory before accessing tensor.shape

    // Use Unified Memory so host and device can both access shape & strides
    CUDA_ERROR_CHECK(cudaMemcpy(tensor.shape, shape, dims * sizeof(int), cudaMemcpyHostToDevice));

    tensor.compute_strides(); // Now safe to compute strides

    // Print directly on the host without explicit copies
    std::cout << "Old Shape: ";
    for (int i = 0; i < dims; i++) std::cout << tensor.shape[i] << " ";
    std::cout << std::endl;

    std::cout << "Old Strides: ";
    for (int i = 0; i < dims; i++) std::cout << tensor.strides[i] << " ";
    std::cout << std::endl;

    // Define a permutation (swap first and second dimensions)
    int h_perm[] = {1, 0, 3, 2};
    int* d_perm;
    CUDA_ERROR_CHECK(cudaMallocManaged(&d_perm, dims * sizeof(int)));  // Use Unified Memory
    CUDA_ERROR_CHECK(cudaMemcpy(d_perm, h_perm, dims * sizeof(int), cudaMemcpyHostToDevice));

    // Launch permute kernel (metadata update)
    permute_kernel<<<1, 1>>>(tensor, d_perm);
    // uncomment for cpu version
    // tensor.permute(h_perm);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // No need to copy back! Host and Device share memory as we used unified memory.
    std::cout << "New Shape: ";
    for (int i = 0; i < dims; i++) std::cout << tensor.shape[i] << " ";
    std::cout << std::endl;

    std::cout << "New Strides: ";
    for (int i = 0; i < dims; i++) std::cout << tensor.strides[i] << " ";
    std::cout << std::endl;

    // Free memory
    tensor.free();
    CUDA_ERROR_CHECK(cudaFree(d_perm));

    return 0;
}