## Day 74
Today we learn more about `ThreadblockShape`, `WarpShape` and `InstuctionShape` that we've been defining to lay the GEMM kernel out in a composable hierarchical sections. At each thread, wartp and thread-block level they compute their own tile-size with higher level of tile sizes being composed from lower level ones.

### Heirarchical Composition

Let's look into how we define a ThreadBlockShape and what is means.

```cpp
// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;    // TensorCore instruction shape
```

#### Thread Block Shape

For a problem size of shape: `M = 4096, N = 2048 and K = 1024`, each thread block would compute the output tile of shape: `128 x 128`  and each time thread block process a tile, it moves 128 values in K dimension before moving on to the next set. 

The matrix A based on our problem size will be of shape: `4096 x 1024` and for a tile size of `128 x 128`, it will be divided into `32 x 8` tiles of shape `128 x 128` each. Similarly for B, which is of shape: `1024 x 2048`, it will be divided into `8 x 16` tiles of shape `128 x 128`. The output is of shape: `4096 x 2048`, so we'd need `32 x 16` thread blocks to run the computation.

While each thread block processes `128` elements in the **K** dimension at once, the full K dimension is `1024`, so each thread block needs to do `8` iterations (`1024÷128 = 8`) along **K** to complete its output tile calculation.

#### Warp Shape

Let's look into the warp shapenow. A warp shape of `64 x 64 x 128` means that each warp (32 threads) will be responsible for computing an output of shape `64 x 64`, a thread block (with multiple warps) divides its `128 × 128` tile into four `64 × 64` subtiles (`2` in **M** dimension × `2` in **N** dimension). As before it has to complete `8` iterations along the **K** dimension to compute the `64 x 64` subtile.

#### Instruction Shape

This is the size of a single MMA instruction that gets executed on a *TensorCore* Each *TensorCore* processes a `8 x 8 x 32` chunk of data. This means that each warp's `64 x 64 x 128` tile gets divided into `8 x 8 x 4` instruction level tiles. For each subtile corresponding to a warp, a *TensorCore* has to execute `8 x 8 x 4 = 256` MMA instructions, which is super efficient!

This fine-grained division allows tensor cores to efficiently process matrix multiplications in parallel. The tensor cores can execute these MMA operations very efficiently compared to traditional CUDA cores.

The beauty of this hierarchical tiling structure (thread block → warp → instruction) is that it maps perfectly to the GPU's architecture and memory hierarchy, maximizing computational throughput and minimizing data movement!

### Next Steps

We look into adding Conv2d kernels and also writing Custom Epilogue ops so that we can do fusion for more complicated kernels like Attention.