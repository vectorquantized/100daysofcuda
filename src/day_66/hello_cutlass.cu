#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main(int argc, char* argv[]) {
    cutlass::half_t x = 2.25_hf;
    std::cout << "x: " << x << std::endl;
    return 0;
}