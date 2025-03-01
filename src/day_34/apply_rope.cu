#include "../cuda/rope.h"
#include "apply_rope.h"


void apply_rope_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& pos_ids,
    const torch::Tensor& cos,   
    const torch::Tensor& sin,   
    torch::Tensor& Q_out,
    torch::Tensor& K_out,
    c10::optional<at::cuda::CUDAStream> stream) {

    // Get tensor dimensions
    const int B = Q.size(0); // Batch size
    const int H = Q.size(1); // Number of heads
    const int L = Q.size(2); // Sequence length
    const int D = Q.size(3); // Head dimension

    // Create CUDA stream
    cudaStream_t cuda_stream = stream.has_value() ? stream.value().stream() 
                                                  : at::cuda::getCurrentCUDAStream().stream();

    // Verify tensor shapes
    TORCH_CHECK(K.size(0) == B && Q_out.size(0) == B && K_out.size(0) == B, 
                "Inconsistent batch size across tensors");
    TORCH_CHECK(K.size(1) == H && Q_out.size(1) == H && K_out.size(1) == H, 
                "Inconsistent number of heads across tensors");
    TORCH_CHECK(K.size(2) == L && Q_out.size(2) == L && K_out.size(2) == L,  // Fixed N to L
                "Inconsistent sequence length across tensors");
    TORCH_CHECK(K.size(3) == D && Q_out.size(3) == D && K_out.size(3) == D, 
                "Inconsistent head dimension across tensors");
    
    // Verify CUDA tensors
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && Q_out.is_cuda() && K_out.is_cuda(),
                "All tensors must be CUDA tensors");

    // Get data pointers
    const float* q_ptr = Q.data_ptr<float>();
    const float* k_ptr = K.data_ptr<float>();
    const int* pos_ids_ptr = pos_ids.data_ptr<int>();  // Changed to int from float
    const float* cos_ptr = cos.data_ptr<float>();
    const float* sin_ptr = sin.data_ptr<float>();
    float* q_out_ptr = Q_out.data_ptr<float>();
    float* k_out_ptr = K_out.data_ptr<float>();
    
    dim3 threads(std::min(128, (D + 1) / 2));
    dim3 blocks(
        (D / 2 + threads.x - 1) / threads.x,
        L,
        B * H
    );

    RopeContext<int> ctx = {  
        .position_ids = pos_ids_ptr, 
        .batch_size = B,
        .num_heads = H,
        .seq_len = L,
        .head_dim = D, 
    };

    apply_rope<<<blocks, threads, 0, cuda_stream>>>(
        q_ptr, k_ptr, cos_ptr, sin_ptr, ctx, q_out_ptr, k_out_ptr
    );
}