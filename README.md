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
* ~something is wrong with the way I am storing exp_vals. They don't match cpu softmax.~
    * No need to use exp_vals, will tackle this in may be the online version.  
* ~also rescaling could be an issue.~
    * This is only valid for online version.
 
## Day 7 Online Softmax
This is very close to what is done in FlashAttention paper.
But we'll build it step by step. 
Few things to note: 
* Online softmax is not a magic pill or one size fits all. It's needed in specific scenarios. 
    * If matrix size is smallish like 8192 x 8192 then we have a batch size: 8192 and hidden size 8192. 
    * The naive version might end up being faster as online version has branching and we might see divergence.
* The online version is exploting the property that $e^{a + b} = e^{a} \cdot e^{b}$.
* We calculate the norm and max in one pass but we also need to perform rescaling of the norm. Let's see why:
        
`x = [1, 3, 4, 2]`, at index `0`, the value `1` is max, so we could just do:

$$
\text{max} = max_{i} 
$$
$$
\text{norm} = e^{(x_{i} - \text{max})} = e^{(1 - 1)}
$$

At index `1`, $\text{max}$ becomes `3`. The value of the global maximum $\text{max}$ has changed, so the previously calculated $\text{norm}$ needs to be updated.

We could re-write:

$$
e^{(x_{i} - \text{max})} = e^{(x_{i} - max_{i})} \cdot e^{(max_{i} - \text{max})}
$$

The quantity $e^{(max_{i} - \text{max})}$ becomes the **rescaling factor**.  
In the example above, at index `1$, we update the norm as:

```cpp
norm *= exp(1 - 3)
```
We then add to $\text{norm}$ the exponentiated value:
```cpp
norm += exp(x[i] - max)
```

* The kernel runtimes are as follows:
```
💾 Memory Allocation on Device        :  0.638    ms
💾 Mem copy (cudaMemcpyHostToDevice)  :  88.360   ms
🚀 Kernel execution time              :  95.489   ms
🚀 Online Kernel execution time       :  91.670   ms
🚀 Kernel execution time              :  94.027   ms
🚀 Online Kernel execution time       :  89.229   ms
💾 Mem copy (cudaMemcpyDeviceToHost)  :  106.156  ms
✅ Test Passed! CPU and GPU outputs match.
```

## Day 9 Batched softmax
We are still on softmax, one might wonder why softmax and why so much time on softmax. Softmax offers an opprtunity to go tiled, has reduction operations and has an online version. So far we've done tiled, online and naive. We wanted to do batched version to take a step closer to implementing self attention and flash attention. By introducing a batched dimension, we are able to see clear benefits of online softmax. In self attention, the input is of shape: (B, L, D) which we transform to (B, L, h, d) -> (B, h, L, d) -> scale(Q)K ->  Softmax(scale(Q)K). Softmax is usually applied on shape: (B, h, L). When B is higher and h is decently higher and L is large, we need online softmax. We've implemented this as:

```cpp
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= B * L) return; 

    int batch_idx = row / L;
    int seq_idx = row % L;
```
We launch `B*L` threads per block and each thread operates on D dimensions. Although this can be improved and is the topic for tomorrow but we clearly see online softmax winning in perf time compared to offline.

Some numbers on `NVIDIA RTX 2000`
```
Mesuring performance characteristics of online vs offline softmax
💾 Memory Allocation on Device        :  0.899    ms
💾 Mem copy (cudaMemcpyHostToDevice)  :  318.910  ms
🚀 Kernel execution time              :  244.755  ms
🚀 Online Kernel execution time       :  206.970  ms
🚀 Kernel execution time              :  250.438  ms
🚀 Online Kernel execution time       :  206.711  ms
💾 Mem copy (cudaMemcpyDeviceToHost)  :  325.269  ms
```

There's a difference of 44 ms between the two kernels, with online version faster than the offline one by ~44 ms or online version is 17.46% faster.




