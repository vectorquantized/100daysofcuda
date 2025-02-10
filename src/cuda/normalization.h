#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <cuda_runtime.h>
#define CEIL_DIV(M, N) ((M + N - 1 ) / N)

template<typename T>
__global__ void layer_norm(const T* __restrict__ input, 
                      T* __restrict__ output, 
                      const T* __restrict__ gamma, 
                      const T* __restrict__ beta,
                      int num_features, T eps) {

    /*Here we do the following:
    Each row is operated on per block. The number of elements num_features 
    could be larger than the block size or the number of threads in a block
    so naturally, each thread operates on multiple elements to calculate the mean and variance.
    We go strided, start with idx = threadIdx.x (of current thread) and go until we can (idx < num_features)
    with a stride of blockDim.x

    We need to fetch the row for a block, the indexing for input and output is the same.
    block_input = input + blockIdx.x * num_features;
    block_output = input + blockIdx.x * num_features;
    We fetch x = block_input[idx] and add it to sum and sum_squared.

    We then in shared memory add the work done by each thread.
    After a barrier sync, we do parallel reduction using a strided pattern we saw before.
    Thread 0 of each block is responsible for calculating the final mean and inverse std dev
    for a row.

    Then we do the layer norm op:
    norm = (x - mean) * inv_std;
    output = norm * gamma[idx] + beta[idx].

    We will allocate gammas, betas and output on kernel launch.

    */ 


    int block = blockIdx.x;
    int tx = threadIdx.x;

    const T* block_input = input + block * num_features;
    T* block_output = output + block * num_features;

    T sum = static_cast<T>(0);
    T sum_squared = static_cast<T>(0);

    // strided loop
    for(int i = tx; i < num_features; i+= blockDim.x) {
        T x = block_input[i];
        sum += x;
        sum_squared += x * x;
    }
    extern __shared__ T smem[];
    T* smem_sum = smem;
    T* smem_sum_squared = smem + blockDim.x;

    smem_sum[tx] = sum;
    smem_sum_squared[tx] = sum_squared;
    __syncthreads();

    for(int stride = blockDim.x / 2 ; stride > 0; stride /= 2) {
        if(tx < stride) {
            smem_sum[tx] += smem_sum[tx + stride];
            smem_sum_squared[tx] += smem_sum_squared[tx + stride];
        }
        __syncthreads();
    }

    T mean = smem_sum[0] / num_features;
    T var = smem_sum_squared[0] / num_features - mean * mean;

    T inv_std = rsqrtf(var + eps);

    __shared__ T smem_mean, smem_inv_std;
    if(tx == 0) {
        smem_mean = mean;
        smem_inv_std = inv_std;
    }
    __syncthreads();
    mean = smem_mean;
    inv_std = smem_inv_std;

    for(int i = tx; i < num_features; i += blockDim.x) {
        T x = block_input[i];
        T norm = (x - mean) * inv_std;
        block_output[i] = norm * gamma[i] + beta[i];
    }


}

#endif // NORMALIZATION_H