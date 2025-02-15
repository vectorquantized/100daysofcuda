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
__global__ void swiglu(const T* __restrict__ up, 
                        const T* __restrict__ gate, 
                        T* __restrict__ output,
                         int B, int L, int H) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < B * L * H) {

        int row = idx % H;
        int col = idx / H;

        int b = row % L;
        int l = row % L;

        int base_idx = b * (L * H) + l * H;
        T x1 = up[base_idx + col];
        T x2 = gate[base_idx + H + col];

        T silu = x2 * sigmoid<T>(x2);

        output[base_idx + col] = x1 * silu;

    }
}

#endif //ACTIVATIONS_H