#ifndef REDUCTION_H
#define REDUCTION_H

#include <cuda_runtime.h>
#define CEIL_DIV(M, N) ((M + N - 1 ) / N)

template<typename T>
__global__ void sum_atomic(const T* __restrict__ input, T* __restrict__ output, T* result, int size) {
    extern __shared__ T smem[];
    int tx = threadIdx.x;
    int gx = tx + blockIdx.x * blockDim.x;

    if (gx < size) {
        smem[tx] = input[gx];
    } else {
        smem[tx] = static_cast<T>(0);
    }

    __syncthreads();

    for(int stride = blockDim.x / 2 ; stride > 0; stride /= 2) {
        if (tx < stride ) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads();
    }
    if(tx == 0) {
        output[blockIdx.x] = smem[0];
        atomicAdd(result, output[blockIdx.x]); 
    }

}

#endif // REDUCTION_H