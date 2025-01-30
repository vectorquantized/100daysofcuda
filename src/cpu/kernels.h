#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include <vector>
#include <type_traits>
// #include <omp.h>

namespace cpu_kernels {

void vector_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
void gemm(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int M, int K, int N);

template<typename T>
T exp_func(const T& value) {
    if constexpr(std::is_same_v<T, float>) {
        return expf(value);
    } else if constexpr(std::is_same_v<T, double>) {
        return exp(value);
    } else if constexpr(std::is_same_v<T, long double>) {
        return expl(value);
    } else {
        static_assert(std::is_floating_point_v<T>, "exp_func requires a floating point type");
    }
}

template<typename T, typename Func>
std::vector<T> for_each(const std::vector<T>& input, Func func) {
    std::vector<T> output(input.size());
    // #pragma omp parallel for
    for(int i=0; i< input.size(); ++i) {
        output[i] = func(input[i]);
    }
    return output;
}

template<typename T>
std::vector<T> softmax(const std::vector<T>& input) {
    auto exponentiated = for_each(input, [](T x) { return exp_func(x);});
    auto normalization = std::accumulate(exponentiated.begin(), exponentiated.end(), static_cast<T>(0));
    auto result = for_each(exponentiated, [normalization](T x) {return x / normalization;});
    return result;
}

}
#endif //CPU_KERNELS_H

