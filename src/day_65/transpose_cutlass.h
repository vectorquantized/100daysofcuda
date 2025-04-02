#ifndef TRANSPOSE_CUTLASS_NAIVE_H
#define TRANSPOSE_CUTLASS_NAIVE_H

#include <torch/extension.h>

torch::Tensor transpose_fast(torch::Tensor A);

#endif // TRANSPOSE_CUTLASS_NAIVE_H