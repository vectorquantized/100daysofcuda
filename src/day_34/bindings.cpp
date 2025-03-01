#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "apply_rope.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_rope", &apply_rope_forward, "Apply rope to Q and K CUDA forward pass", 
        py::arg("query"), py::arg("key"), py::arg("pos_ids"), 
        py::arg("cos"), py::arg("sin"), 
        py::arg("q_out"), py::arg("k_out"), py::arg("stream") = py::none());
}