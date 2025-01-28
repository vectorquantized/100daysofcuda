#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <random>
#include <vector>
#include <fmt/core.h>

namespace cpu_utils {
void init_random_vector(std::vector<float>& array, size_t size, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution (0.0f, 1.0f);
    for(size_t i = 0; i< size; ++i) {
        array[i] = distribution(generator);
    }
}

inline void print_formatted_vector(const std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        fmt::print("{:.2f}{}", vec[i], (i < vec.size() - 1 ? ", " : "\n"));
    }
}

inline bool compare_vectors(const float* vec1, const float* vec2, int size, float epsilon = 1e-4) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(vec1[i] - vec2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}
} //namespace cpu_utils

// Macro definition
#define COMPARE_RESULT(reference, output, size, epsilon)                      \
    do {                                                                      \
        bool validated = cpu_utils::compare_vectors(reference, output, size, epsilon); \
        if (validated) {                                                     \
            std::cout << "Test Passed!" << std::endl;                        \
        } else {                                                             \
            std::cout << "CPU and GPU kernel results don't match" << std::endl; \
        }                                                                    \
    } while (0)

#endif //CPU_UTILS_H
