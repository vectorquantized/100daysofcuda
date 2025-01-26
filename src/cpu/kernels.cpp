#include <iostream>
#include <stdexcept>
#include "cpu/kernels.h"

namespace cpu_kernels {
void vector_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    c.resize(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}
}
