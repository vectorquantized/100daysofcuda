/**
 * Structure to hold attention configuration parameters
 */
namespace host {
template<typename T>
 struct AttentionContext {
    const int B;           // Batch size
    const int H;           // Number of attention heads
    const int N;           // Sequence length
    const int D;           // Head dimension
    const float scale;     // 1/sqrt(D) - Pre-computed scaling factor
    const bool is_causal;  // Whether to apply causal masking
    
    // Constructor for convenient initialization
    __host__ AttentionContext(int batch_size, int num_heads, int seq_len, int head_dim, bool causal = false)
        : B(batch_size), 
          H(num_heads), 
          N(seq_len), 
          D(head_dim),
          scale(static_cast<T>(1.0f / sqrtf(static_cast<float>(head_dim)))),
          is_causal(causal) {}
};
} // namespace host

namespace device {

/**
 * Structure to hold block processing indices and dimensions
 */
 struct BlockIndices {
    int batch_idx;       // Current batch
    int head_idx;        // Current attention head
    int row_block_idx;   // Block index for query rows
    int col_block_idx;   // Block index for key/value columns
    int tid;             // Thread ID within block
    int valid_rows;      // Number of valid rows in current block
    int valid_cols;      // Number of valid columns in current block
    int row_start;       // Starting row index for current block
    int col_start;       // Starting column index for current block
    
    // Constructor for convenient initialization
    __device__ BlockIndices(
        int b_idx, int h_idx, int rb_idx, int cb_idx, int t_id, int block_size_m
    ) : batch_idx(b_idx), 
        head_idx(h_idx), 
        row_block_idx(rb_idx), 
        col_block_idx(cb_idx), 
        tid(t_id),
        valid_rows(0),
        valid_cols(0),
        row_start(rb_idx * block_size_m),
        col_start(cb_idx) {}
};

/**
 * Initialize the softmax accumulators and output values
 */
 template<typename T, int ROWS_PER_WARP, int D_VALUE>
 __device__ void initialize_accumulators(
     T m_i[ROWS_PER_WARP],
     T l_i[ROWS_PER_WARP],
     T o_i[ROWS_PER_WARP][D_VALUE]
 ) {
     #pragma unroll
     for (int i = 0; i < ROWS_PER_WARP; i++) {
         m_i[i] = -INFINITY;
         l_i[i] = static_cast<T>(0);
         #pragma unroll
         for (int d = 0; d < D_VALUE; d++) {
             o_i[i][d] = static_cast<T>(0);
         }
     }
 }
} // namespace device