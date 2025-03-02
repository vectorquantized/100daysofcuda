#ifndef ROPE_KERNEL_H
#define ROPE_KERNEL_H

#include <cuda_runtime.h>

template<typename IndexType>
struct RopeContext {
    const int batch_size;
    const int seq_len;
    const int num_heads;
    const int head_dim;
    const IndexType* __restrict__ position_ids;
};

template<typename T, typename IndexType>
__global__ void apply_rope(
    const T* __restrict__ Q, 
    const T* __restrict__ K, 
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    const RopeContext<IndexType> ctx, 
    T* __restrict__ Q_out, 
    T* __restrict__ K_out) {

        int dim_offset = threadIdx.x + blockIdx.x * blockDim.x; // x dim indexes into the head dim
        int seq_idx = blockIdx.y; // y dim gives us the row.
        int batch_head_idx = blockIdx.z; // z dim contains both head and batch together.

        int batch_idx = batch_head_idx / ctx.num_heads;
        int head_idx = batch_head_idx % ctx.num_heads;

        if (batch_idx < ctx.batch_size && 
            head_idx < ctx.num_heads && 
            seq_idx < ctx.seq_len &&
            dim_offset * 2 + 1 < ctx.head_dim) {
                int base_idx =((batch_idx * ctx.seq_len  + seq_idx) * ctx.num_heads + head_idx) * ctx.head_dim;
                int pos_id = ctx.position_ids[batch_idx * ctx.seq_len + seq_idx];
            
                const T* cos_pos = cos + pos_id * ctx.head_dim / 2;
                const T* sin_pos = sin + pos_id * ctx.head_dim / 2;
                int dim_idx = base_idx + dim_offset * 2;

                T q_even = Q[dim_idx];
                T q_odd = Q[dim_idx + 1];
                T k_even = K[dim_idx];
                T k_odd = K[dim_idx + 1];

                T cos_val = cos_pos[dim_offset];
                T sin_val = sin_pos[dim_offset];

                Q_out[dim_idx] = q_even * cos_val - q_odd * sin_val;
                Q_out[dim_idx + 1] = q_odd * cos_val + q_even * sin_val;

                K_out[dim_idx] = k_even * cos_val - k_odd * sin_val;
                K_out[dim_idx + 1] = k_odd * cos_val + k_even * sin_val;
        }
}

#endif // ROPE_KERNEL_H