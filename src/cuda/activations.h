#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#define TILE_WIDTH 16

#include <cuda_runtime.h>
#include <cmath>

template<typename T>
__device__ T sigmoid(T x) {
    return T(1) / (T(1) + exp(-x));
}

template<typename T>
__global__ void swiglu(const T* __restrict__ up, 
            const T* __restrict__ gate, 
            T*       __restrict__ output,
            int B, int L, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * L * H) 
    {
        // decode (b,l,h) from idx
        int h   = idx % H;
        int tmp = idx / H;
        int l   = tmp % L;
        int b   = tmp / L;

        // base offset in [B,L,H]
        int base_idx = b*(L*H) + l*H + h;

        T x1 = up[base_idx];
        T x2 = gate[base_idx];
        
        T silu = x2 * sigmoid(x2);

        output[base_idx] = x1*silu;
    }
}

#endif //ACTIVATIONS_H