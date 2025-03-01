
#include "../cuda/attention/flash_attention.h"


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void flash_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O,
    const bool is_causal,
    c10::optional<at::cuda::CUDAStream> stream = c10::nullopt) {
    // Get tensor dimensions
    const int B = Q.size(0); // Batch size
    const int H = Q.size(1); // Number of heads
    const int N = Q.size(2); // Sequence length
    const int D = Q.size(3); // Head dimension

    // Create CUDA stream
    cudaStream_t cuda_stream = stream.has_value() ? stream.value().stream() 
                                                  : at::cuda::getCurrentCUDAStream().stream();
    
    // Constants for block sizes (can be tuned based on GPU architecture)
    constexpr int BLOCK_SIZE_M = 64;  // Number of query rows per block
    constexpr int BLOCK_SIZE_N = 64;  // Number of key/value rows per block
    constexpr int THREADS_PER_BLOCK = 256;
    
    // Verify tensor shapes
    TORCH_CHECK(K.size(0) == B && V.size(0) == B && O.size(0) == B, 
                "Inconsistent batch size across tensors");
    TORCH_CHECK(K.size(1) == H && V.size(1) == H && O.size(1) == H, 
                "Inconsistent number of heads across tensors");
    TORCH_CHECK(K.size(2) == N && V.size(2) == N && O.size(2) == N, 
                "Inconsistent sequence length across tensors");
    TORCH_CHECK(K.size(3) == D && V.size(3) == D && O.size(3) == D, 
                "Inconsistent head dimension across tensors");
                
    // Verify CUDA tensors
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda(),
                "All tensors must be CUDA tensors");
                
    // Check tensor type and dispatch to float implementation for now
  
    // Create the attention context
    host::AttentionContext<float> ctx(B, H, N, D, is_causal);
    
    // Calculate grid dimensions
    dim3 grid(ctx.B, ctx.H, (ctx.N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    dim3 block(THREADS_PER_BLOCK);
    
    // Get data pointers
    const float* q_ptr = Q.data_ptr<float>();
    const float* k_ptr = K.data_ptr<float>();
    const float* v_ptr = V.data_ptr<float>();
    float* o_ptr = O.data_ptr<float>();
    
    // Launch kernel with appropriate template parameters based on D
    if (ctx.D <= 32) {
        device::flash_attention_kernel<float, BLOCK_SIZE_M, BLOCK_SIZE_N, 32><<<grid, block, 0, cuda_stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, ctx);
    } else if (ctx.D <= 64) {
        device::flash_attention_kernel<float, BLOCK_SIZE_M, BLOCK_SIZE_N, 64><<<grid, block, 0, cuda_stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, ctx);
    } else if (ctx.D <= 128) {
        device::flash_attention_kernel<float, BLOCK_SIZE_M, BLOCK_SIZE_N, 128><<<grid, block, 0, cuda_stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, ctx);
    } else {
        // For larger dimensions, default to 128 and handle with loops inside kernel
        device::flash_attention_kernel<float, BLOCK_SIZE_M, BLOCK_SIZE_N, 128><<<grid, block, 0, cuda_stream>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, ctx);
    }
    
}