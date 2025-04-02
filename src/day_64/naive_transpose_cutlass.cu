#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../cuda/cutlass/transpose.h"

#define TILE_WIDTH 16

//------------------------------------------------------------------------------
// Host function: transpose_naive
//
// This function sets up the environment for the transpose kernel. It performs the
// following key steps:
// 1. Constructs tensor shapes and layouts for both source and destination matrices.
// 2. Wraps raw GPU pointers in tensor objects using make_tensor and the given layout.
// 3. Creates a transposed view of the output to perform an in-place transpose.
// 4. Tiles the global tensors into blocks (tiles) for efficient parallel processing.
// 5. Defines a thread-level partition (thread layout) to further subdivide each tile.
// 6. Configures grid and block dimensions based on the tiled tensor shapes.
// 7. Launches the transpose kernel with the appropriate configuration.
//------------------------------------------------------------------------------

torch::Tensor transpose_naive(torch::Tensor A) {

  TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
  TORCH_CHECK(A.ndimension() == 2, "A must be a 2D tensor.");
  int M = A.size(0);
  int N = A.size(1);
  auto B = torch::zeros({N, M}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
  // Use AT_DISPATCH to handle different data types
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "transpose_naive", [&] {
    // Create the shape of the input tensor with dimensions (M, N).
    auto tensor_shape = cute::make_shape(M, N);
    auto tensor_shape_trans = cute::make_shape(N, M);
    
    // Define layout for input tensor (row-major)
    auto gmemLayoutA = cute::make_layout(tensor_shape, cute::LayoutRight{});
    auto gmemLayoutB = cute::make_layout(tensor_shape_trans, cute::LayoutRight{});
    
    // Explicitly cast to the correct type pointer before creating gmem_ptr
    // This avoids the void* dereference issue
    auto tensor_A = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<scalar_t*>(A.data_ptr())), 
        gmemLayoutA);
    
    // Create transposed view of output with explicit type
    auto gmemLayoutBT = cute::make_layout(tensor_shape, cute::GenColMajor{});
    auto tensor_BT = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<scalar_t*>(B.data_ptr())), 
        gmemLayoutBT);
    
    // Define block (tile) sizes
    using bM = cute::Int<64>;
    using bN = cute::Int<64>;
    
    // Create block shape
    auto block_shape = cute::make_shape(bM{}, bN{});
    
    // Tiled division with proper layouts
    auto tiled_tensor_A = cute::tiled_divide(tensor_A, block_shape);
    auto tiled_tensor_BT = cute::tiled_divide(tensor_BT, block_shape);
    
    // Define thread layouts
    auto threadLayoutA = cute::make_layout(
        cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), 
        cute::LayoutRight{});
    auto threadLayoutB = cute::make_layout(
        cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), 
        cute::LayoutRight{});
    
    // Configure grid and block dimensions
    dim3 gridDim(
        cute::size<1>(tiled_tensor_A),  // First dimension 
        cute::size<2>(tiled_tensor_A)   // Second dimension
    );
    dim3 blockDim(cute::size(threadLayoutA));
    
    // Launch the transpose kernel
    transpose_kernel_naive<<<gridDim, blockDim>>>(
        tiled_tensor_A, tiled_tensor_BT,
        threadLayoutA, threadLayoutB);
  });
  
  return B;
};