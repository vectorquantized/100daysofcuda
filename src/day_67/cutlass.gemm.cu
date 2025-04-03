#include <cuda_runtime.h>
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
    if(deviceProp.major == 8) {
        using AmpereConfig = GemmConfig<cutlass::arch::Sm80>;
        status = run_gemm<AmpereConfig>(M, N, K, alpha, beta);
    } else if (deviceProp.major ==7 && deviceProp.minor == 5) {
        using AmpereConfig = GemmConfig<cutlass::arch::Sm75>;
        status = run_gemm<AmpereConfig>(M, N, K, alpha, beta);
    } else {
        std::cerr << "Unsupported compute capability: "
                  << deviceProp.major << "." << deviceProp.minor << std::endl;
        return -1;
    }
    
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