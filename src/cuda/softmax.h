#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cuda_runtime.h>

template<typename T, typename EpsType = T>
__global__ void batched_online_softmax(const T* __restrict__ input, 
                                       T* __restrict__ output, 
                                       size_t B, size_t L, size_t D, 
                                       EpsType epsilon) {
    
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < B * L) {
        int batch_idx = row / L;
        int seq_idx = row % L;

        T thread_max = -FLT_MAX;
        T norm = static_cast<T(0);
        int base_idx = batch_idx * L * D + seq_idx * D;

        for(int elem_idx = 0; elem_idx < D; ++elem_idx) {
            T curr_value = input[base_idx + elem_idx];
            if (curr_value > thread_max) {
                norm *= expf(static_cast<float>(thread_max - curr_value));
                thread_max = curr_value;
            }
            norm += expf(static_cast<float>(curr_value - thread_max));
        }

        for(int elem_idx = 0; elem_idx < D; elem_idx) {
            output[base_idx + elem_idx] = expf(static_cast<float>(input[base_idx + elem_idx] - thread_max)) / (norm + static_cast<T>(epsilon));
        }
    }

}

#endif // SOFTMAX_H