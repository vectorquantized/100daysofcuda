#include "../attention/attention_helpers.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace device {
/**
 * Flash Attention Implementation
 */
 template<typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int D_VALUE>
 __global__ void flash_attention_kernel(
     const T* __restrict__ Q,
     const T* __restrict__ K,
     const T* __restrict__ V,
     T* __restrict__ O,
     const host::AttentionContext<T> ctx
 ) {
     // Thread identification
     const int tid = threadIdx.x;
     const int lane_idx = tid % 32;
     const int warp_idx = tid / 32;
     const int warp_count = blockDim.x / 32;
     
     // Each warp handles specific rows
     const int rows_per_warp = BLOCK_SIZE_M / warp_count;
     const int warp_row_offset = warp_idx * rows_per_warp;
     
     // Define shared memory for tiled computation
     __shared__ T s_Q[BLOCK_SIZE_M][D_VALUE];
     __shared__ T s_K[BLOCK_SIZE_N][D_VALUE];
     __shared__ T s_V[BLOCK_SIZE_N][D_VALUE];
     
     // Thread-local accumulators for online softmax
     T m_i[rows_per_warp];      // Max value for numerical stability
     T l_i[rows_per_warp];      // Sum of exponentials
     T o_i[rows_per_warp][D_VALUE]; // Output accumulators
     
     // Initialize all accumulators
     initialize_accumulators<T, rows_per_warp, D_VALUE>(m_i, l_i, o_i);
     
     // Get the tile's warp group
     cg::thread_block block = cg::this_thread_block();
     cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
     
     // Process all key/value blocks
     for (int col_block_idx = 0; col_block_idx < ctx.N; col_block_idx += BLOCK_SIZE_N) {
         // Create block indices struct
         BlockIndices indices(blockIdx.x, blockIdx.y, blockIdx.z, col_block_idx, tid, BLOCK_SIZE_M);
         
         // Skip if using causal mask and this block is entirely in the future
         if (ctx.is_causal && indices.col_start > indices.row_start + BLOCK_SIZE_M - 1) {
             continue;
         }
         
         // Load QKV data into shared memory
         load_qkv_blocks<T, BLOCK_SIZE_M, BLOCK_SIZE_N, D_VALUE>(
             Q, K, V, s_Q, s_K, s_V, ctx, indices
         );
         
         // Make sure all data is loaded before computing
         __syncthreads();
         
         // Each warp processes its assigned query rows
         for (int local_row = 0; local_row < rows_per_warp; local_row++) {
             // Find absolute row index
             int row_idx = warp_row_offset + local_row;
             
             // Skip processing if the row is out of bounds
             if (row_idx >= indices.valid_rows) continue;
             
             // Compute attention scores for this row against all valid columns
             for (int col_idx = 0; col_idx < indices.valid_cols; col_idx++) {
                 // Skip if using causal mask and col_idx is after row_idx
                 int abs_row = indices.row_start + row_idx;
                 int abs_col = indices.col_start + col_idx;
                 if (ctx.is_causal && abs_col > abs_row) continue;
                 
                 // Calculate QK^T dot product
                 float qk = compute_dot_product<T, D_VALUE>(
                     s_Q[row_idx], s_K[col_idx], ctx.scale, warp
                 );
                 
                 // Update softmax state and output vectors
                 update_softmax_state<T, D_VALUE>(
                     m_i[local_row], l_i[local_row], o_i[local_row], 
                     qk, s_V[col_idx], lane_idx
                 );
             }
         }
         
         // Make sure all computations are done before loading next block
         __syncthreads();
     }
     
     // Write final output to global memory
     const int batch_head_offset = blockIdx.x * ctx.H * ctx.N * ctx.D + blockIdx.y * ctx.N * ctx.D;
     const int row_start = blockIdx.z * BLOCK_SIZE_M;
     
     for (int local_row = 0; local_row < rows_per_warp; local_row++) {
         int row_idx = warp_row_offset + local_row;
         
         // Skip if row is out of bounds
         if (row_start + row_idx >= ctx.N) continue;
         
         // Each thread writes its part of the output
         for (int d = lane_idx; d < D_VALUE; d += 32) {
             int o_pos = batch_head_offset + (row_start + row_idx) * ctx.D + d;
             O[o_pos] = o_i[local_row][d];
         }
     }
 }
} // namespace device