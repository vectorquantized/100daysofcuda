#ifndef TRANSPOSE_CUTLASS_NAIVE_H
#define TRANSPOSE_CUTLASS_NAIVE_H

#include <torch/extension.h>

torch::Tensor transpose_naive(torch::Tensor A);

#endif // TRANSPOSE_CUTLASS_NAIVE_H