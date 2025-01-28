#include <iostream>
#include <stdexcept>
#include "cpu/kernels.h"
#include "csrc/timing_utils.h"

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

void gemm(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int M, int K, int N) {
    TIMED_CPU_FUNCTION();
    c.resize(M * N);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float p_value = 0.0f;
            for (int i = 0; i < K; ++i) {
                p_value += a[row * K + i] * b[i * N + col];
            }
            c[row * N + col] = p_value;
        }
    }
}
}
