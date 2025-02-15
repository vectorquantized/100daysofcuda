#ifndef ACTICATIONS_H
#define ACTIVATIONS_H

#define TILE_WIDTH 16

#include <cuda_runtime.h>
#include <cmath>

template<typename T>
__device__ T sigmoid(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

template<typename T>
__global__ void swilglu(const T* __restrict__ up, const T* __restrict__ gate, T* __restrict__ output, int B, int L, int D) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < B * L) {

        int row = idx % D;
        int col = idx / D;

        int b = row % L;
        int l = row % L;

        int base_idx = b * (L * D) + l * D;
        T x1 = up[base_idx + col];
        T x2 = gate[in_base_idx + D + col];

        T silu = x2 * sigmoid<T>(x2);

        output[base_idx + col] = x1 * silu;

    }
}

#endif //ACTIVATIONS_H