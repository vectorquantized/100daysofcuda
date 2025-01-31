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

template<typename T>
std::vector<T> softmax(const std::vector<T>& input, size_t M, size_t N, float epsilon) {
    std::vector<T> output(input.size());
    
    // Process each row separately
    for(size_t i = 0; i < M; i++) {
        // Find max in this row
        T max_val = -std::numeric_limits<T>::max();
        for(size_t j = 0; j < N; j++) {
            max_val = std::max(max_val, input[i * N + j]);
        }
        
        // Compute exp(x - max) and sum
        T sum_exp = 0;
        for(size_t j = 0; j < N; j++) {
            T exp_val = exp_func(input[i * N + j] - max_val);
            output[i * N + j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for(size_t j = 0; j < N; j++) {
            output[i * N + j] /= (sum_exp + epsilon);
        }
    }
    
    return output;
}

}
#endif //CPU_KERNELS_H

