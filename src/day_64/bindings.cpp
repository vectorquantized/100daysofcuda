#include <torch/extension.h>
#include "naive_transpose_cutlass.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_slow", &transpose_naive, "Transpose Naive forward pass.");
}