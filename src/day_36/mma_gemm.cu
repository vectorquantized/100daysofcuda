#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include "../cuda/gemm.h"
using namespace nvcuda;

// Error checking macros
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
        exit(1);
    }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
        exit(1);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
        exit(1);
    }
}


// Matrix dimensions
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 32768

// WMMA dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Conversion kernels
__global__ void convertFp32ToFp16(half *out, const float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void convertFp16ToFp32(float *out, const half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

int main(int argc, char* argv[]) {
    float *a_fp32;
    float *b_fp32;
    half *a_fp16;
    half *b_fp16;

    float *c;
    float *c_cublas;
    float *c_wmma;

    float *c_host_cublas;
    float *c_host_wmma;
    
    curandGenerator_t gen;
    cublasHandle_t cublasHandle;
    
    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    
    cudaEvent_t startcublas;
    cudaEvent_t stopcublas;
    
    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA));
    
    cudaErrCheck(cudaEventCreate(&startcublas));
    cudaErrCheck(cudaEventCreate(&stopcublas));
    
    
    cublasErrCheck(cublasCreate(&cublasHandle));
    
    // Use tensor cores
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
    
    cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

    c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

    curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
    convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

    curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
    
    curandErrCheck(curandDestroyGenerator(gen));
    
    cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;


    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
    
    // First: using our templated WMMA kernel
    dim3 gridDim;
    dim3 blockDim;
 
    // blockDim.x must be a multiple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    
    printf("Running with templated WMMA...\n");
    cudaErrCheck(cudaEventRecord(startWMMA));
    tensor_core::gemm_mma<WMMA_M, WMMA_K, WMMA_N, half, float>
        <<<gridDim, blockDim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K);
    cudaErrCheck(cudaEventRecord(stopWMMA));
    cudaErrCheck(cudaEventSynchronize(stopWMMA));

    // Now using cuBLAS
    printf("Running with cuBLAS...\n");
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        MATRIX_M, MATRIX_N, MATRIX_K,
        &alpha,
        a_fp32, CUDA_R_32F, MATRIX_M,  // Changed from a_fp16 and CUDA_R_16F
        b_fp32, CUDA_R_32F, MATRIX_K,  // Changed from b_fp16 and CUDA_R_16F
        &beta,
        c_cublas, CUDA_R_32F, MATRIX_M,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));

    // Error checking
    printf("\nChecking results...\n");
    cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
       float v1 = c_host_wmma[i];
       float v2 = c_host_cublas[i];
       float diff  = fabs(v1 - v2);
       float relative_err = diff / v2;
       float eps = 1e-3;
       if ((relative_err >= eps)) {
          errors++;
          if (errors < 10) printf("%f %f\n", v1, v2);
       }
    }
    
    if (errors > 0) {
       printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
    }
    else {
       printf("Results verified: cublas and WMMA agree.\n\n");
       float wmmaTime;
       float cublasTime;
       cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
       cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
       printf("templated wmma took %fms\n", wmmaTime);
       printf("cublas took %fms\n", cublasTime);
    }
    
    // Cleanup
    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));

    cudaErrCheck(cudaEventDestroy(startcublas));             
    cudaErrCheck(cudaEventDestroy(stopcublas));
    
    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));

    cudaErrCheck(cudaFree(c));
    cudaErrCheck(cudaFree(c_cublas));
    cudaErrCheck(cudaFree(c_wmma));
    
    free(c_host_cublas);
    free(c_host_wmma);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}