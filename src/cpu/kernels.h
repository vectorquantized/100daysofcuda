#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include <vector>

namespace cpu_kernels {
void vector_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
void gemm(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int M, int K, int N);
}
#endif //CPU_KERNELS_H

