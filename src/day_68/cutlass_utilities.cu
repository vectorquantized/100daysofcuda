#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include "../cuda/cutlass/gemm.h"

int main(int argc, char* argv[]) {

    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);  // device 0
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Compute capability: " 
              << deviceProp.major << "." << deviceProp.minor << std::endl;


    int M = 1024;
    int N = 512;
    int K = 256;

    float alpha = 1.0f;
    float beta = 0.0;
    cutlass::Status status;

    using ELementType = cutlass::half_t;
    using LayoutType = cutlass::layout::ColumnMajor;
    using Config = GemmConfig<cutlass::arch::Sm80>;

    if(deviceProp.major == 8) {
        using Config = GemmConfig<cutlass::arch::Sm80>;
        // status = run_gemm<AmpereConfig>(M, N, K, alpha, beta);
    } else if (deviceProp.major ==7 && deviceProp.minor == 5) {
        using Config = GemmConfig<cutlass::arch::Sm75>;
        // status = run_gemm<TuringConfig>(M, N, K, alpha, beta);
    } else {
        std::cerr << "Unsupported compute capability: "
                  << deviceProp.major << "." << deviceProp.minor << std::endl;
        return -1;
    }

    uint64_t seed = 2080;

    // Gaussian random distribution
    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;

    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one = 0;

    GaussianInitializer<Config::ElementA, Config::LayoutA> init_A(seed, mean, stddev, bits_less_than_one);
    GaussianInitializer<Config::ElementB, Config::LayoutB> init_B(seed * 2019, mean, stddev, bits_less_than_one);
    GaussianInitializer<Config::ElementC, Config::LayoutC> init_C(seed * 1993, mean, stddev, bits_less_than_one);
    
    status = run_gemm<Config, 
    GaussianInitializer<Config::ElementA, Config::LayoutA>,
    GaussianInitializer<Config::ElementB, Config::LayoutB>,
    GaussianInitializer<Config::ElementC, Config::LayoutC>
    >(M, N, K, alpha, beta, init_A, init_B, init_C);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    if(status != cutlass::Status::kSuccess) {
        std::cout << "GEMM ERROR: " << static_cast<int>(status) << std::endl;
        return -1;
    }

    return 0;
}