#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <random>
#include <vector>
#include <fmt/core.h>
#include <execution>  // C++17 Parallel STL

namespace cpu_utils {
    
template <typename Allocator = std::allocator<float>>
void init_random_vector(std::vector<float, Allocator>& array, size_t size, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    // Ensure vector is resized properly
    if (array.size() != size) {
        array.resize(size);
    }

    // Generate random values directly into array
    std::generate(std::execution::par_unseq, array.begin(), array.end(),
                  [&]() { return distribution(generator); });
}

inline void print_formatted_vector(const std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        fmt::print("{:.2f}{}", vec[i], (i < vec.size() - 1 ? ", " : "\n"));
    }
}

inline bool compare_vectors(const float* vec1, const float* vec2, int size, float epsilon = 1e-4) {
    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_index = -1;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(vec1[i] - vec2[i]);
        if (diff > epsilon) {
            mismatches++;
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_index = i;
            }
        }
    }

    if (mismatches == 0) {
        return true;  // All values match within tolerance
    } else {
        std::cout << "❌ CPU and GPU results **do not match**!\n"
                  << "Total mismatches: " << mismatches << "/" << size << "\n"
                  << "Max difference: " << max_diff << " at index " << max_diff_index << "\n"
                  << "CPU: " << vec1[max_diff_index] << " | GPU: " << vec2[max_diff_index] << std::endl;
        return false;
    }
}

inline void print_vectors(const float* b_ref, const float* b_h, int size) {
    float max_diff = 0.0f;
    int max_diff_index = -1;

    for (size_t i = 0; i < size; ++i) {
    float diff = fabs(b_h[i] - b_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_index = i;
        }
    }

    std::cout << "🔍 Max Difference: " << max_diff 
            << " at index " << max_diff_index << "\n"
            << "GPU: " << b_h[max_diff_index] 
            << " | CPU: " << b_ref[max_diff_index] 
            << std::endl;
}

template <typename T, typename GeneratorFunc>
void initialize_batched_matrices_col_major(
    T* data,
    int batch_count,
    int rows, int cols,
    int ld, // leading dimension
    int batch_stride,
    GeneratorFunc generator // data generator function.
) {
    
    for (int b_idx = 0; b_idx < batch_count; b_idx++) {
        for (int col_idx = 0; col_idx < cols; col_idx++) {
            for (int row_idx = 0; row_idx < rows; row_idx++) {
                int idx = b_idx * batch_stride + col_idx * ld +  row_idx; // column major
                data[idx] = generator(b_idx, row_idx, col_idx);
            }
        }
    }
}

std::vector<size_t> generate_random_permutation(size_t size) {
    std::vector<size_t> permutation(size);

    // fill the permutation vector with the values 0, 1, 2, 3, ..., size-1
    std::iota(permutation.begin(), permutation.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    std::shuffle(permutation.begin(), permutation.end(), g);
    
    return permutation;
}

} //namespace cpu_utils

// Macro definition
#define COMPARE_RESULT(reference, output, size, epsilon)                   \
    do {                                                                   \
        bool validated = cpu_utils::compare_vectors(reference, output, size, epsilon); \
        if (validated) {                                                   \
            std::cout << "✅ Test Passed! CPU and GPU outputs match.\n";   \
        } else {                                                           \
            std::cout << "❌ Test Failed! See above for mismatch details.\n"; \
        }                                                                  \
    } while (0)

#endif //CPU_UTILS_H
