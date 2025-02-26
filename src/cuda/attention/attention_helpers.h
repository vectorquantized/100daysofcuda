#include "../attention/utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace device {

/**
* Load Q, K, V blocks into shared memory
*/
template<typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int D_VALUE>
__device__ void load_qkv_blocks(
   const T* __restrict__ Q,
   const T* __restrict__ K,
   const T* __restrict__ V,
   T s_Q[BLOCK_SIZE_M][D_VALUE],
   T s_K[BLOCK_SIZE_N][D_VALUE],
   T s_V[BLOCK_SIZE_N][D_VALUE],
   const host::AttentionContext<T>& ctx,
   BlockIndices& indices
) {
   // Calculate batch and head offset for indexing
   const int batch_head_offset = indices.batch_idx * ctx.H * ctx.N * ctx.D + 
                                 indices.head_idx * ctx.N * ctx.D;
   
   // Get number of valid rows/cols in current blocks (handle boundary)
   indices.valid_rows = min(BLOCK_SIZE_M, ctx.N - indices.row_start);
   indices.valid_cols = min(BLOCK_SIZE_N, ctx.N - indices.col_start);
   
   // Load Q block to shared memory (each thread loads multiple elements)
   for (int row_offset = 0; row_offset < indices.valid_rows; row_offset += blockDim.x / D_VALUE) {
       int row_idx = row_offset + indices.tid / D_VALUE;
       if (row_idx < indices.valid_rows) {
           int col_idx = indices.tid % D_VALUE;
           int q_pos = batch_head_offset + (indices.row_start + row_idx) * ctx.D + col_idx;
           s_Q[row_idx][col_idx] = Q[q_pos];
       }
   }
   
   // Load K block to shared memory
   for (int col_offset = 0; col_offset < indices.valid_cols; col_offset += blockDim.x / D_VALUE) {
       int col_idx = col_offset + indices.tid / D_VALUE;
       if (col_idx < indices.valid_cols) {
           int d_idx = indices.tid % D_VALUE;
           int k_pos = batch_head_offset + (indices.col_start + col_idx) * ctx.D + d_idx;
           s_K[col_idx][d_idx] = K[k_pos];
       }
   }
   
   // Load V block to shared memory
   for (int col_offset = 0; col_offset < indices.valid_cols; col_offset += blockDim.x / D_VALUE) {
       int col_idx = col_offset + indices.tid / D_VALUE;
       if (col_idx < indices.valid_cols) {
           int d_idx = indices.tid % D_VALUE;
           int v_pos = batch_head_offset + (indices.col_start + col_idx) * ctx.D + d_idx;
           s_V[col_idx][d_idx] = V[v_pos];
       }
   }
}



/**
 * Compute dot product using warp-level parallelism
 */
 template<typename T, int D_VALUE>
 __device__ T compute_dot_product(
     const T* q_row,
     const T* k_row,
     const T scale,
     const cg::thread_block_tile<32>& warp
 ) {
     const int lane_idx = threadIdx.x % 32;
     T qk_sum = static_cast<T>(0);
     
     // Each thread computes partial dot product
     #pragma unroll
     for (int d = lane_idx; d < D_VALUE; d += 32) {
         qk_sum += q_row[d] * k_row[d];
     }
     
     // Reduce within the warp
     #pragma unroll
     for (int offset = 16; offset > 0; offset /= 2) {
         qk_sum += warp.shfl_down(qk_sum, offset);
     }
     
     // Apply scaling
     if (lane_idx == 0) {
         qk_sum *= scale;
     }
     
     // Broadcast to all threads
     return warp.shfl(qk_sum, 0);
 }


 /**
 * Update online softmax state and output for a single element
 */
template<typename T, int D_VALUE>
__device__ void update_softmax_state(
    T& m_prev,
    T& l_prev,
    T* o_prev,
    const T qk,
    const T* v_row,
    const int lane_idx
) {
    // Calculate new max value for numerical stability
    T m_new = max(m_prev, qk);
    T l_new;
    
    // Compute new denominator
    if (isinf(m_prev)) {
        // First valid attention score
        l_new = static_cast<T>(1);
    } else {
        // Scale old sum and add new value
        l_new = l_prev * exp(m_prev - m_new) + exp(qk - m_new);
    }
    
    // Apply online softmax update rule
    for (int d = lane_idx; d < D_VALUE; d += 32) {
        if (isinf(m_prev)) {
            // First entry, just set the value
            o_prev[d] = v_row[d];
        } else {
            // Update running weighted sum
            T scale_prev = exp(m_prev - m_new) / l_new;
            T scale_new = exp(qk - m_new) / l_new;
            o_prev[d] = o_prev[d] * scale_prev + v_row[d] * scale_new;
        }
    }
    
    // Update softmax tracking variables
    m_prev = m_new;
    l_prev = l_new;
}

} //namespace device