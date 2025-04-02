#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cuda/gemm.h"
#include "../cuda/cuda_utils.h"
#include "../cuda/cutlass/transpose.h"
#include "transpose_cutlass.h"

torch::Tensor transpose_fast(torch::Tensor A) {
    TORCH_CHECK(A.device().is_cuda() , "A must be a cuda tensor");
    TORCH_CHECK(A.ndimension() == 2, "A must be a 2D tensor.");
    int M = A.size(0);
    int N = A.size(1);
    auto B = torch::zeros({N, M}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "transpose_fast", [&] {
      // Create the shape of the input tensor with dimensions (M, N).
      auto tensor_shape = cute::make_shape(M, N);
      auto tensor_shape_trans = cute::make_shape(N, M);
      
      // Define layout for input tensor (row-major)
      auto gmemLayoutA = cute::make_layout(tensor_shape, cute::LayoutRight{});
      auto gmemLayoutB = cute::make_layout(tensor_shape_trans, cute::LayoutRight{});
      
      // Explicitly cast to the correct type pointer before creating gmem_ptr
      auto tensor_A = cute::make_tensor(
          cute::make_gmem_ptr(static_cast<scalar_t*>(A.data_ptr())), 
          gmemLayoutA);
      
      auto tensor_B = cute::make_tensor(
          cute::make_gmem_ptr(static_cast<scalar_t*>(B.data_ptr())), 
          gmemLayoutB);
      
      // Define block (tile) sizes
      using bM = cute::Int<64>;
      using bN = cute::Int<64>;
      
      // Create block shapes
      auto block_shape_A = cute::make_shape(bM{}, bN{});
      auto block_shape_B = cute::make_shape(bN{}, bM{});
      
      // Tiled division with proper layouts
      auto tiled_tensor_A = cute::tiled_divide(tensor_A, block_shape_A);
      auto tiled_tensor_B = cute::tiled_divide(tensor_B, block_shape_B);
      
      auto tile_shape_A = cute::make_layout(block_shape_A, cute::LayoutRight{});
      auto tile_shape_B = cute::make_layout(block_shape_B, cute::LayoutRight{});
  
      auto smemLayoutA = tile_shape_A;
      auto smemLayoutB = cute::composition(smemLayoutA, tile_shape_B);
      auto smemLayoutA_swizzle = cute::composition(cute::Swizzle<5, 0, 5>{}, tile_shape_A);
      auto smemLayoutB_swizzle = cute::composition(smemLayoutA_swizzle, tile_shape_B);
  
      // Define thread layouts
      auto threadLayoutA = cute::make_layout(
          cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), 
          cute::LayoutRight{});
      auto threadLayoutB = cute::make_layout(
          cute::make_shape(cute::Int<8>{}, cute::Int<32>{}), 
          cute::LayoutRight{});
      
      // Calculate shared memory size - using the correct variable name
      size_t smem_size = sizeof(SharedStorageTranspose<scalar_t, decltype(smemLayoutA_swizzle)>);
      
      // Configure grid and block dimensions
      dim3 gridDim(
          cute::size<1>(tiled_tensor_A),
          cute::size<2>(tiled_tensor_A)
      );
      dim3 blockDim(cute::size(threadLayoutA));
      
      // Launch the transpose kernel - make sure the kernel name matches your definition
      transpose_kernel_smem<<<gridDim, blockDim, smem_size>>>(
          tiled_tensor_A, tiled_tensor_B, smemLayoutA, threadLayoutA,
          smemLayoutB, threadLayoutB);
    });
    
    return B;
  };