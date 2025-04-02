#include <torch/extension.h>
#include "transpose_cutlass.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose", &transpose_fast, "Transpose with smem forward pass.");
}