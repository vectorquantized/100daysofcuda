#ifndef BATCHED_CONV2D_H
#define BATCHED_CONV2D_H

#include <torch/extension.h>

torch::Tensor batched_conv2d_forward(torch::Tensor input, torch::Tensor kernel);

#endif // BATCHED_CONV2D_H
