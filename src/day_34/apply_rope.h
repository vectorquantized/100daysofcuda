#ifndef APPLY_ROPE_H
#define APPLY_ROPE_H

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void apply_rope_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& pos_ids,
    const torch::Tensor& cos,   
    const torch::Tensor& sin,   
    torch::Tensor& Q_out,
    torch::Tensor& K_out,
    c10::optional<at::cuda::CUDAStream> stream = c10::nullopt);

#endif // APPLY_ROPE_H
