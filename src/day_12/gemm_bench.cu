#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

torch::Tensor batched_gemm_forward(torch::Tensor A, torch::Tensor B, float scale) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.ndimension() == 3 && B.ndimension() == 3, "A and B must be 3D tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match.");
    TORCH_CHECK(A.size(2) == B.size(1), "A's columns must match B's rows.");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));

    // Leading dimensions and strides
    int64_t lda = K;
    int64_t ldb = N;
    int64_t ldc = N;
    int64_t stridea = M * K;
    int64_t strideb = K * N;
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
        std::cerr << "No suitable cuBLASLt algorithm found!\n";
        return torch::empty({}); 
    }

   
    float alpha = 1.0f;
    float beta = 0.0f;

    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasLtMatmul(
        ltHandle,
        operationDesc,
        &alpha,
        A.data_ptr<float>(), // A matrix
        Adesc,
        B.data_ptr<float>(), // B matrix
        Bdesc,
        &beta,
        C.data_ptr<float>(), // C matrix (input)
        Cdesc,
        C.data_ptr<float>(), // C matrix (output)
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

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_fast", &batched_gemm_forward, "Batched CUBLAS GEMM forward pass");
}