
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>  
#include <iomanip>
#include <vector>

#define CEIL_DIV(M, N) ((M + N - 1 ) / N)      

#define CUDA_ERROR_CHECK(call)  { \
    call; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error after kernel launch in file '%s' at line %i: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error on device synchronization in file '%s' at line %i: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

class CudaEventTimer {
public:
    CudaEventTimer(const char* event_name):
    event_name_(event_name) {
        // std::cout << "Constructor called for: " << event_name_ << std::endl;
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_);
    }

    ~CudaEventTimer() {
        // std::cout << "Destructor called for: " << event_name_ << std::endl;
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        
        // Find longest operation name for alignment
        constexpr size_t col_width = 40;
        
        std::cout << std::left << std::setw(col_width) << event_name_ 
                  << ":  " << std::fixed << std::setprecision(3) 
                  << std::setw(8) << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    const char* event_name_;
    cudaEvent_t start_, stop_;
};

template <typename T>
struct CudaPinnedAllocator {
    using value_type = T;

    T* allocate(std::size_t num) {
        T* ptr;
        CUDA_ERROR_CHECK(cudaMallocHost((void**)&ptr, num * sizeof(T)));
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        CUDA_ERROR_CHECK(cudaFreeHost(ptr));
    }
};

template <typename T>
class PinnedVector : public std::vector<T, CudaPinnedAllocator<T>> {
public:
    // Inherits constructors from vector
    using std::vector<T, CudaPinnedAllocator<T>>::vector;
    
    // Convert to standard vector
    std::vector<T> to_std_vector() const {
        return std::vector<T>(this->begin(), this->end());
    }

    // Implicit conversion operator for automatic conversion
    operator std::vector<T>() const {
        return to_std_vector();
    }
};

#define TIMED_CUDA_FUNCTION() CudaEventTimer(__FUNCTION__)

#define TIMED_CUDA_BLOCK(BLOCK_NAME) CudaEventTimer timer##__LINE__(BLOCK_NAME)

#endif // CUDA_UTILS_H