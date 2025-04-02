#ifndef CUTLASS_TRANSPOSE_H
#define CUTLASS_TRANSPOSE_H
// Prevent multiple inclusion of this header file

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// Thrust is used here for managing device (GPU) and host (CPU) vectors.
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUTLASS and CuTe headers: CUTLASS provides core utilities and numeric types
// while CuTe provides tensor abstractions and architecture-specific support.
#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "../../cuda/cutlass/util.h"

template <class Element, class SmemLayout> struct SharedStorageTranspose {
    cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                        cutlass::detail::alignment_for_swizzle(SmemLayout{})>
        smem;
  };

template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    transpose_kernel_smem(TensorS const S, TensorD const D,
                        SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                        SmemLayoutD const smemLayoutD, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // two different views of smem
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // (bN, bM)

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tSsS = local_partition(sS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x);
  Tensor tDsD = local_partition(sD, tD, threadIdx.x);

  cute::copy(tSgS, tSsS); // LDGSTS

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  cute::copy(tDsD, tDgD);
}


// This kernel performs a naive transpose operation on a matrix tile.
// It uses CuTe tensors to partition global memory (gmem) into thread-local 
// tiles, copies data into a temporary register memory, and then writes it back.
// The function calls like make_tensor, local_partition, and copy are part of
// the CuTe library which abstracts tensor operations for GPU kernels.
template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transpose_kernel_naive(TensorS const S, TensorD const DT,
                       ThreadLayoutS const tS, ThreadLayoutD const tD) {
  // Define Element type from the source tensor's value type.
  using Element = typename TensorS::value_type;

  // Extract the current block's global tile from the source tensor.
  // make_coord(_, _) creates a full range for the first two dimensions.
  // blockIdx.x and blockIdx.y index the current block.
  auto gS = S(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y);   // Tile shape: (bM, bN)
  
  // Extract the current block's global tile from the destination tensor view.
  auto gDT = DT(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y); // Tile shape: (bM, bN)

  // Partition the global source tile among threads according to tS.
  // local_partition splits the tensor tile for the thread identified by threadIdx.x.
  auto tSgS = cute::local_partition(gS, tS, threadIdx.x); // Local tile: (ThrValM, ThrValN)
  
  // Partition the global destination tile among threads according to tD.
  auto tDgDT = cute::local_partition(gDT, tD, threadIdx.x);

  // Create a temporary register memory tensor that has the same shape as tSgS.
  auto rmem = cute::make_tensor_like(tSgS);

  // Copy data from the thread-local source tile into register memory.
  cute::copy(tSgS, rmem);
  
  // Copy the data from register memory to the thread-local destination tile.
  cute::copy(rmem, tDgDT);
}

#endif // CUTLASS_TRANSPOSE_H