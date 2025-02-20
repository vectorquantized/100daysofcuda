#include <torch/extension.h>
#include "batched_conv2d.h"

// Register the function in a separate file
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2D", &batched_conv2d_forward, "2D convolution CUDA forward pass");
}