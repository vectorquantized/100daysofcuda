#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vector>
#include <fmt/core.h>

void init_random_vector(std::vector<float>& array, size_t size, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution (0.0f, 1.0f);
    array.resize(size);
    for(size_t i = 0; i< size; ++i) {
        array[i] = distribution(generator);
    }
}

void print_formatted_vector(const std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        fmt::print("{:.2f}{}", vec[i], (i < vec.size() - 1 ? ", " : "\n"));
    }
}

#endif //UTILS_H