#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "../cuda/cublas_manager.h"


template<typename T>
void cublas_lt_matmul_optimized(const T* A, const T* B, T* C, int batch_size, int M, int K, int N, CublasLtManager& manager, cudaStream_t stream = nullptr, int strideb = -1) {
    // Leading dimensions and strides
    int64_t lda = K;
    int64_t ldb = N;
    int64_t ldc = N;
    int64_t stridea = M * K;
    if (strideb < 0) {
        strideb = static_cast<int64_t>(K) * N;
    }
    int64_t stridec = M * N;

    // Use the persistent handle, workspace, and stream
    cublasLtHandle_t ltHandle = manager.getHandle();
    void* workspace = manager.getWorkspace();
    size_t workspace_size = manager.getWorkspaceSize();
    bool should_sync = false;
    if (stream == nullptr) {
        stream = manager.getStream();
        should_sync = true;
    }

    // Create Matmul Descriptor (still per-call unless shapes are fixed)
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create Matrix Layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, ldc);

    int32_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // Set Batch Attributes
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));

    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));

    // Set up the preference
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspace_size, sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristicResult[1]; 
    int returnedResults = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 1, heuristicResult, &returnedResults);

    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        throw std::runtime_error("No suitable cuBLASLt algorithm found!");
    }

    T alpha = static_cast<T>(1);
    T beta  = static_cast<T>(0);

    cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        (const void*)A, Adesc,
        (const void*)B, Bdesc,
        &beta,
        (const void*)C, Cdesc,
        (void*)C, Cdesc,
        &heuristicResult[0].algo,
        workspace,
        workspace_size,
        stream);

    if (should_sync) {
        cudaStreamSynchronize(stream);
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}

template<typename T>
void cublas_lt_matmul(const T* A, const T* B, T* C, int batch_size, int M, int K, int N, int strideb = -1) {

    // Leading dimensions and strides
    int64_t lda = K;
    int64_t ldb = N;
    int64_t ldc = N;
    int64_t stridea = M * K;
    if (strideb < 0) {
        strideb = static_cast<int64_t>(K) * N;
    }
    int64_t stridec = M * N;

    // Create cuBLASLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // Create Matmul Descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_N, transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create Matrix Layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, ldc);

    // Set the matrix order to row-major for each descriptor
    int32_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // Set Batch Attributes
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));

    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));

    size_t workspace_size = 1 * 1024 * 1024;
    void* workspace;
    cudaMalloc(&workspace, workspace_size);

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspace_size, sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristicResult[1]; 
    int returnedResults = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 
        1, 
        heuristicResult, &returnedResults);

    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        throw std::runtime_error("No suitable cuBLASLt algorithm found!");
    }

   
    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);

    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        (const void*)A, 
        Adesc,
        (const void*)B, 
        Bdesc,
        &beta,
        (const void*)C,  // “input C” if needed
        Cdesc,
        (void*)C,        // “output D”
        Cdesc,
        &heuristicResult[0].algo,
        workspace,
        workspace_size,
        stream);

   
    cudaStreamSynchronize(stream);

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
    cudaFree(workspace);
    cudaStreamDestroy(stream);

}