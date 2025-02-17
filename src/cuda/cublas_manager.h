// cublas_lt_manager.h
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdexcept>

class CublasLtManager {
public:
    CublasLtManager(size_t workspace_size_bytes = (1 * 1024 * 1024)) 
        : workspace_size(workspace_size_bytes), workspace(nullptr) {
        // Create cuBLASLt handle
        if(cublasLtCreate(&ltHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLASLt handle");
        }
        // Allocate workspace once
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if(err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate workspace");
        }
        // Create a persistent CUDA stream
        err = cudaStreamCreate(&stream);
        if(err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~CublasLtManager() {
        if (workspace) cudaFree(workspace);
        cublasLtDestroy(ltHandle);
        cudaStreamDestroy(stream);
    }

    cublasLtHandle_t getHandle() const { return ltHandle; }
    void* getWorkspace() const { return workspace; }
    size_t getWorkspaceSize() const { return workspace_size; }
    cudaStream_t getStream() const { return stream; }

private:
    cublasLtHandle_t ltHandle;
    void* workspace;
    size_t workspace_size;
    cudaStream_t stream;
};