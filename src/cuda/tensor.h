#ifndef TEN_TENSOR_H
#define TEN_TENSOR_H

#include <cuda_runtime.h>
#include "cuda/cuda_utils.h"

// The definition of this structure is inspired by: https://github.com/1y33/100Days/blob/main/day04/layerNorm.cu#L6
namespace ten {

struct Tensor {
    float* data;
    int size;
    int* shape;   // Shape of the tensor
    int* strides; // Strides for correct memory indexing
    int dims;     // Number of dimensions

    __host__ void allocate(int N, int D) {
        size = N;
        dims = D;
        CUDA_ERROR_CHECK(cudaMalloc((void**)&data, size * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMallocManaged(&shape, dims * sizeof(int)));
        CUDA_ERROR_CHECK(cudaMallocManaged(&strides, dims * sizeof(int)));
    }

    __host__ void compute_strides() {
        strides[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    __host__ void free() {
        CUDA_ERROR_CHECK(cudaFree(data));
        CUDA_ERROR_CHECK(cudaFree(shape));
        CUDA_ERROR_CHECK(cudaFree(strides));
        data = nullptr;
        shape = nullptr;
        strides = nullptr;
    }

    // N-dimensional indexing on device
    __device__ float& operator()(const int* indices) {
        int flat_index = 0;
        for (int i = 0; i < dims; ++i) {
            flat_index += indices[i] * strides[i];
        }
        return data[flat_index];
    }

    // Variadic operator() for indexing
    template<typename... Indices>
    __host__ __device__ float& operator()(Indices... indices) {
        int idx_array[] = {indices...};
        return (*this)(idx_array);
    }

    // In-place transposition on device
    __device__ void transpose(const int* perm) {
        int temp_shape[10];   // TODO: change this, although 10 works for now.
        int temp_strides[10]; // Buffer for transposed metadata

        // Copy data to temporary arrays and permute
        for (int i = 0; i < dims; ++i) {
            temp_shape[i] = shape[perm[i]];
            temp_strides[i] = strides[perm[i]];
        }

        // Copy back to shape and strides
        for (int i = 0; i < dims; ++i) {
            shape[i] = temp_shape[i];
            strides[i] = temp_strides[i];
        }
    }

    // 1D access for raw pointer-style indexing
    __device__ float& operator[](int index) {
        return data[index];
    }
};

}

#endif //TEN_TENSOR_H