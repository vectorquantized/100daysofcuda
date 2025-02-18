// main.cpp

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#include <mma.h>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/tensor_view_io.h"

int main(int argc, char const **argv) {

  // Problem size: C = A x B
  int M = 1024;  // number of rows in A and C
  int N = 1024;  // number of columns in B and C
  int K = 4096;  // number of columns in A and rows in B

  // Create CUTLASS host tensors for matrices A, B, and C.
  // Using RowMajor layout.
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_A({M, K});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_B({K, N});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_C({M, N});

  // Initialize A and B with random data.
  for (int i = 0; i < tensor_A.size(); ++i) {
    tensor_A.host_data()[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < tensor_B.size(); ++i) {
    tensor_B.host_data()[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Copy host data to device.
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();

  // Define the GEMM operator type using CUTLASS.
  // This instantiation uses float as the data type and RowMajor layout for all matrices.
  using Gemm = cutlass::gemm::device::Gemm<
      float, cutlass::layout::RowMajor,   // Element, Layout for A
      float, cutlass::layout::RowMajor,   // Element, Layout for B
      float, cutlass::layout::RowMajor,   // Element, Layout for C
      float>;                             // Element for the epilogue (accumulation)

  // Create a GEMM operator instance.
  Gemm gemm_op;

  // Define the arguments for the GEMM operation.
  typename Gemm::Arguments arguments{
      {M, N, K},                                   // Problem size (M, N, K)
      {tensor_A.device_data(), tensor_A.layout()},   // Tensor A info
      {tensor_B.device_data(), tensor_B.layout()},   // Tensor B info
      {tensor_C.device_data(), tensor_C.layout()},   // Tensor C info (source matrix)
      {tensor_C.device_data(), tensor_C.layout()},   // Tensor D info (destination matrix)
      {1.0f, 0.0f}                                   // Scalars used in the epilogue: alpha, beta
  };

  // Allocate workspace if required.
  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t err = cudaMalloc(&workspace, workspace_size);
    if (err != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }
  }

  // Initialize the GEMM operator.
  cutlass::Status status = gemm_op.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM initialization failed: " << int(status) << std::endl;
    return -1;
  }

  // Launch the GEMM operator.
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM execution failed: " << int(status) << std::endl;
    return -1;
  }

  // Copy the result from device back to host.
  tensor_C.sync_host();

  // (Optional) Verify correctness with a reference GEMM.
  // For brevity, this step is omitted. You can use cutlass::reference::host::Gemm
  // to compute a CPU reference and compare the results.

  // Free workspace.
  if (workspace) {
    cudaFree(workspace);
  }

  std::cout << "CUTLASS GEMM executed successfully." << std::endl;
  return 0;
}