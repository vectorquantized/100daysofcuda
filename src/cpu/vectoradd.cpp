#include <iostream>
#include <stdexcept>
#include "utils.h"

void add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    c.resize(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    size_t size = 1024; // we'd like to add the vectors of this size.
    unsigned int baseSeed = 42;
    std::vector<float> a;
    std::vector<float> b;
    init_random_vector(a, size, baseSeed);
    init_random_vector(b, size, baseSeed + 1);
    std::vector<float> c;
    c.resize(size);
    add(a, b, c);
    print_formatted_vector(c);
    return 0;
}