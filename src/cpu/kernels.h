#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include <vector>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <cfloat>
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
std::vector<T> softmax(const std::vector<T>& input, size_t M, size_t N, size_t TILE_WIDTH, T epsilon) {
    std::vector<T> output(input.size());
    
    std::vector<T> row_max(M, -std::numeric_limits<T>::max());
    std::vector<T> row_sum(M, static_cast<T>(0));
    std::vector<std::vector<T>> exp_values(M, std::vector<T>(N, static_cast<T>(0)));

    // Step 1: Compute max value per row
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < N; ++col) {
            row_max[row] = std::max(row_max[row], input[row * N + col]);
        }
    }

    // Step 2: Compute exp(x - max) per row and store in shared memory
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < N; ++col) {
            exp_values[row][col] = exp_func(input[row * N + col] - row_max[row]);
        }
    }

    // Step 3: Compute sum of exponentials per tile and rescale
    T global_max = *std::max_element(row_max.begin(), row_max.end());
    
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < N; ++col) {
            row_sum[row] += exp_values[row][col];
        }
        row_sum[row] *= exp_func(row_max[row] - global_max);  // ✅ Rescale sum
    }

    // Step 4: Normalize
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < N; ++col) {
            output[row * N + col] = (exp_values[row][col] * exp_func(row_max[row] - global_max)) / (row_sum[row] + epsilon);
        }
    }

    return output;
}

template<typename T>
std::vector<T> softmax_exact(const std::vector<T>& input, size_t M, size_t N, float epsilon) {
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

template <typename T>
void online_softmax(const std::vector<T>& input, std::vector<T>& output, size_t M, size_t N, T epsilon) {

    for (int row = 0; row < M; ++row) {
        T row_max = static_cast<T>(-FLT_MAX);
        T norm = static_cast<T>(0);

        // Step 1: Find the max value for numerical stability
        for (size_t col = 0; col < N; ++col) {
            T curr_value = input[row * N + col];
            if (curr_value > row_max) {
                norm *= std::exp(row_max - curr_value);
                row_max = curr_value;
            }
            norm += std::exp(curr_value - row_max);
        }

        // Step 2: Compute softmax values
        for (size_t col = 0; col < N; ++col) {
            T value = std::exp(input[row * N + col] - row_max);
            output[row * N + col] = value / (norm + epsilon);
        }
    }
}

template <typename T>
void batched_online_softmax(const std::vector<T>& input, std::vector<T>& output, 
                           size_t B, size_t L, size_t D, T epsilon) {
   
    size_t total_elements = B * L * D;
    
    for (size_t output_idx = 0; output_idx < total_elements; ++output_idx) {
      
        int batch_idx = output_idx / (L * D);
        int seq_idx = (output_idx / D) % L;
        
        if (batch_idx < B && seq_idx < L) {  // Same check as GPU
            T thread_max = static_cast<T>(-FLT_MAX);
            T norm = static_cast<T>(0);
            
           
            size_t base_idx = batch_idx * L * D + seq_idx * D;
            
         
            for (size_t elem_idx = 0; elem_idx < D; ++elem_idx) {
                T curr_value = input[base_idx + elem_idx];
                if (curr_value > thread_max) {
                    norm *= std::exp(thread_max - curr_value);
                    thread_max = curr_value;
                }
                norm += std::exp(curr_value - thread_max);
            }
            
       
            for (size_t elem_idx = 0; elem_idx < D; ++elem_idx) {
                output[base_idx + elem_idx] = 
                    std::exp(input[base_idx + elem_idx] - thread_max) / (norm + epsilon);
            }
        }
    }
}
}

#endif //CPU_KERNELS_H

