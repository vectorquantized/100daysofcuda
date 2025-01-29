
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>                                            

#define CUDA_ERROR_CHECK(call)  { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i.%s \n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    }}

#define TIMED_CUDA_FUNCTION() CudaEventTimer(__FUNCTION__)
#define TIMED_CUDA_BLOCK(BLOCK_NAME) CudaEventTimer(BLOCK_NAME)

class CudaEventTimer {

public:
    CudaEventTimer(const char* function_name):
    function_name_(function_name) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }

    ~CudaEventTimer() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        std::cout << "CUDA function " << function_name_ << " finished in " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);

    }
private:
    const char* function_name_;
    cudaEvent_t start_, stop_;
};

#endif // CUDA_UTILS_H