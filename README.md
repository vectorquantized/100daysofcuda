# 100daysofcuda
Learn, code, build using CUDA
All started with joining the discord:  https://discord.gg/4Tg4TkJQzE and getting inspired by everyone.

### Setup
Mentor: https://github.com/hkproj

### Instructions
Checkout: https://github.com/hkproj/100-days-of-cuda

### Why do this?
Want to be a stellar cuda programmer.

### Day 1 VECTOR ADD
Starting Day 1 with a simple kernel. Should perform VectorAdd.
Instructions to run:
```
mkdir build
cd build
cmake ..
make day_1
./day_1
```

### Day 2 GEMM
Day 2 commences with matrix multiplication, baseic version.
We are essentially taking:
```
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float p_value = 0.0f;
            for (int i = 0; i < K; ++i) {
                p_value += a[row * K + i] * b[i * N + col];
            }
            c[row * N + col] = p_value;
        }
    }
```
and utilizing the SIMT architecture of GPUs to parallelize it. The outer loops essentially get scheduled across the SMs and everything else in the basic version remains the same.

Instructions to run:
```
mkdir build
cd build
cmake ..
make day_2
./day_2
```
### Day 3 TILED GEMM

![FlashAttention Banner](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/assets/flashattn_banner.jpg)

<p align="center">Source: <a href="https://github.com/Dao-AILab/flash-attention">Dao-AILab / FlashAttention</a></p>

As seen from the figure SRAM throughput is > 10x HBM. Let's see if we can put this to good use or not. We'll not code Flash Attention as of yet, the figure explains very clearly the difference in throughput of SRAM vs HBM and that's why I used it.

On day 3 we give Tiled GEMM a refresher. In a nutshell we want to perform multiple floating point ops on a particular chunk of data that we transferred in shared memory from HBM and maximize data re-use.

1. Load a tile of matrix A and B from global memory into shared memory
2.	Perform multiple floating-point operations (FLOPs) using this shared memory data, maximizing reuse.
3.	Load the next tile and repeat until the full matrix multiplication is done.

This minimizes global memory accesses and maximizes arithmetic intensity (FLOPs per byte transferred).
Note: If a TILE_WIDTH × TILE_WIDTH tile is loaded, it gets used TILE_WIDTH times before we move to the next tile.
```
mkdir build
cd build
cmake ..
make day_3
./day_3
```

## Day 5 Softmax Kernel
Softmax basic kernel
The basic kernel works and is correct. We don't need to profile it to come to a conclusion that it is slow, reading the kernel should lay out the memory access patterns.
* Needs to be tiled. It is memory bound at this point, we're reading too much from HBM
* The cpu kernel could use OpenMP, couldn't add it to the build system, will figure it out.
* GPU kernel makes an assumption that rows >> cols, which may not be the case, actually, it isn't as Batch size is always << hidden dims.
* Need to extend it to 3D Tensors of Shape: (B, L, D)


## Day 5 Extending Softmax Kernel
Tiled version
* Observed that numerical precision of basic kernel is better compared tiled version (possibly due to reductions in tiled version?)
* Also observed minimal speed differences between the two kernels.
* Lack of speed needs investigation and perhaps better test cases.

## Day 6 Debugging Softmax Tiled Kernel
* row sum and row max match the cpu version.
* something is wrong with the way I am storing exp_vals. They don't match cpu softmax.
* also rescaling could be an issue.
