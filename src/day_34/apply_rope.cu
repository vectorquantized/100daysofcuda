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

    const int B = Q.size(0); // Batch size
    const int L = Q.size(1); // Sequence length
    const int H = Q.size(2); // Number of heads
    const int D = Q.size(3); // Head dimension

    cudaStream_t cuda_stream = stream.has_value() ? stream.value().stream() 
                                                  : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(K.size(0) == B && Q_out.size(0) == B && K_out.size(0) == B, 
                "Inconsistent batch size across tensors");
    TORCH_CHECK(K.size(1) == L && Q_out.size(1) == L && K_out.size(1) == L, 
                "Inconsistent number of heads across tensors");
    TORCH_CHECK(K.size(2) == H && Q_out.size(2) == H && K_out.size(2) == H,
                "Inconsistent sequence length across tensors");
    TORCH_CHECK(K.size(3) == D && Q_out.size(3) == D && K_out.size(3) == D, 
                "Inconsistent head dimension across tensors");
    
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && Q_out.is_cuda() && K_out.is_cuda(),
                "All tensors must be CUDA tensors");

    const float* q_ptr = Q.data_ptr<float>();
    const float* k_ptr = K.data_ptr<float>();
    const int* pos_ids_ptr = pos_ids.data_ptr<int>();
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
        .batch_size = B,
        .seq_len = L,
        .num_heads = H,
        .head_dim = D,
        .position_ids = pos_ids_ptr, 
    };

    apply_rope<<<blocks, threads, 0, cuda_stream>>>(
        q_ptr, k_ptr, cos_ptr, sin_ptr, ctx, q_out_ptr, k_out_ptr
    );
}