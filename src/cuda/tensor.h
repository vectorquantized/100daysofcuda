#ifndef TEN_TENSOR_H
#define TEN_TENSOR_H

#include <cuda_runtime.h>
#include "cuda/cuda_utils.h"

// The definition of this structure is inspired by: https://github.com/1y33/100Days/blob/main/day04/layerNorm.cu#L6
namespace ten {
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
        CUDA_ERROR_CHECK(cudaFree(data));
        data = nullptr;
    }
    __device__ float& operator[] (int i) {
        // add error checking, if needed.
        return data[i];
    }
};
}

#endif //TEN_TENSOR_H