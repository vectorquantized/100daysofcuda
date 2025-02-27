## Day 31 Flash Attention Kernel
On Day 30 we coded Flash Attention Kernel. It's pretty involved to code and get right but the algorithm is pretty simple to understand. At the heart of it lies tiled GEMM (using shared memory) between the query and transposed key matrix and the online softmax implementation that is fused into the kernel. Let's understand what's happening in Flash Attention Kernel to build up to the actual implementation.

### Multi-Head Attention (MHA)
Multi-Head Attention (MHA) is a way to compute the attention scores for each location in the query by looking at the Key and Value matrices. The implementation of MHA is pretty straightforward and the python code looks as follows:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv_fused = nn.Linear(cfg.input_dim, 3 * cfg.hidden_dim, bias = False)
        self.n_heads = cfg.num_heads
        assert(cfg.hidden_dim % cfg.num_heads == 0, "Hidden dimension should be a multiple of number of heads")
        self.model_dim = cfg.hidden_dim // cfg.num_heads
        self.scale = torch.rsqrt(self.model_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias = False)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for MHA
        
        Args:
            input: torch.Tensor of shape: (B, N, E)
        Returns:
            output torch.Tensor of shape: (B, N, O)
        """
        qkv = self.qkv_fused(input)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # Each is of shape: (B, N, D)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        scores = (q * scale) @ k.transpose(2, 3) # shape: (B, H, N, N)
        weights = F.softmax(scores, dim=-1) # shape: (B, H, N, N)
        attention_values = weights @ v # shape: (B, H, N, D)
        attention_values = rearrange(attention_values, 'b d n d -> b n (h d)', h=self.n_heads)
        output = self.out(attention_values)
        return output
```

This is a basic implementation of MHA in PyTorch. Given that we want to write custom CUDA kernels, we assume that we have a custom compiler written for ourselves and it converts the Python code to a CUDA kernel. Assume that the compiler is super simple and looks at each operation looks up the corresponding operation in C++ and calls the binding's wrapper which in turn calls the CUDA kernel. So, we'd then be required to have matrix multiplication and softmax implementations available to us in order to run this code on GPU. 

We'd need matmul kernel for:

```python
scores = (q * scale) @ k.transpose(2, 3) # shape: (B, H, N, N)
attention_values = weights @ v # shape: (B, H, N, D)
```
Note: We ignore the fused qkv and output linear layers for now, and we scale `Q` before performing `Q @ K^T` to prevent numerical instability. Without this, the dot product could grow very large, leading to issues when computing softmax, as exponentiating large numbers can cause overflow.

We'd need the softmax kernel for:
```python
weights = F.softmax(scores, dim=-1) # shape: (B, H, N, N)
```
The algorithm looks like follows:

![Naive Attention Kernel](./basic_attention_impl.png)

Standard attention implementations materialize the matrices `attention_values` $(S)$ and `weights` $(P)$ to HBM, which takes $O(N^2)$ memory, often $ N >> d $ making this kernel memory bound.

#### Flash Attention

##### Introduction

To address the above issues [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) introduces an efficient way of implementing the kernel which makes the kernel compute bound. Let's see how:

First let's discuss online softmax, on day 7 we implemented online softmax, but let's just revisit it. We exploit $e^{a + b} = e^{a} \cdot e^{b}$ property of the exponential function. The basic idea is that we keep a running value of max and norm, if the value of max changes (that is, we encounter a greater value than the current max) then we change the max and rescale the norm. The value by which we rescale the norm is given by: $e^{(max_{i} - \text{max})}$, as:

$$
e^{(x_{i} - \text{max})} = e^{(x_{i} - max_{i})} \cdot e^{(max_{i} - \text{max})}
$$

The authors then propose a fused version (enabled due to tiling) and note that:

*"Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM."*

The outline of the algorithm looks as follows:

![Flash Attention implementation](./flash_attention_impl.png)

##### Kernel Design spec

Our goal is to design a kernel that:
1.	Avoids excessive memory accesses to HBM.
2.	Efficiently loads and computes QK^T in shared memory.
3.	Applies softmax in a numerically stable, online fashion.
4.	Computes the final weighted sum with V without explicitly materializing intermediate matrices.

Inorder to implement this kernel, we'd need a way to load the blocks of Q, K and V matrices in the shared memory then compute their dot product, apply softmax and compute dot product with value block. So, let's see the functions that we could implement:

* `load_qkv_blocks` will load $Q$, $K$ and $V$ in shared memory by block.
* `compute_dot_product` will compute $QK^T$
* A function that computes online softmax, `update_softmax_state`
Note that we perform the softmax statistics update and multiplication with V in one go as follows:
```cpp
T scale_prev = exp(m_prev - m_new) / l_new;
T scale_new = exp(qk - m_new) / l_new;
o_prev[d] = o_prev[d] * scale_prev + v_row[d] * scale_new;
```

###### Load the Q, K, V Blocks in shared memory
Let's look at the function , its signature is as follows:
```cpp
template<typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int D_VALUE>
__device__ void load_qkv_blocks(
   const T* __restrict__ Q,
   const T* __restrict__ K,
   const T* __restrict__ V,
   T s_Q[BLOCK_SIZE_M][D_VALUE],
   T s_K[BLOCK_SIZE_N][D_VALUE],
   T s_V[BLOCK_SIZE_N][D_VALUE],
   const host::AttentionContext<T>& ctx,
   BlockIndices& indices
)
```
We've templatized our implementation. We pass in the `Q`, `K` and `V` arrays and we also pass in the 2D arrays in shared memory. There's an additional `AttentionContext` variable that is passed that contains metadata such as `B`, `H`, `N`, `D`, `scale` and `is_causal`. BlockIndices encapculate:
```cpp
    int batch_idx;       // Current batch
    int head_idx;        // Current attention head
    int row_block_idx;   // Block index for query rows
    int col_block_idx;   // Block index for key/value columns
    int tid;             // Thread ID within block
    int valid_rows;      // Number of valid rows in current block
    int valid_cols;      // Number of valid columns in current block
    int row_start;       // Starting row index for current block
    int col_start;       // Starting column index for current block
```
We then calculate the valid rows and valid cols as follows:

```cpp
    // Get number of valid rows/cols in current blocks (handle boundary)
   indices.valid_rows = min(BLOCK_SIZE_M, ctx.N - indices.row_start);
   indices.valid_cols = min(BLOCK_SIZE_N, ctx.N - indices.col_start);
```
We load a block of `Q`, `K` and `V` based on the values of `valid_rows` (for `Q`) and `valid_cols` (for `K` and `V`).
The loading is done as follows for `Q` (Note that we're loading rows of Q):

```cpp
// Load Q block to shared memory (each thread loads multiple elements)
   for (int row_offset = 0; row_offset < indices.valid_rows; row_offset += blockDim.x / D_VALUE) {
       int row_idx = row_offset + indices.tid / D_VALUE;
       if (row_idx < indices.valid_rows) {
           int col_idx = indices.tid % D_VALUE;
           int q_pos = batch_head_offset + (indices.row_start + row_idx) * ctx.D + col_idx;
           s_Q[row_idx][col_idx] = Q[q_pos];
       }
   }
```
Each thread loads a multiple elements of Q, iterating over rows using row_offset. The `col_idx` ensures we distribute across the depth dimension. So, after the loop has finished we'd have a block of `Q` of shape: $B_r \times d$ loaded in shared memory.

We'd load the corresponding columns of `K`:

```cpp
// Load K block to shared memory
   for (int col_offset = 0; col_offset < indices.valid_cols; col_offset += blockDim.x / D_VALUE) {
       int col_idx = col_offset + indices.tid / D_VALUE;
       if (col_idx < indices.valid_cols) {
           int d_idx = indices.tid % D_VALUE;
           int k_pos = batch_head_offset + (indices.col_start + col_idx) * ctx.D + d_idx;
           s_K[col_idx][d_idx] = K[k_pos];
       }
   }
```

We do the same for V. The observation here is that for the $QK^T$ computation:

* `Q` is on the "left side" (providing rows)
* `K` is on the "right side" (providing columns due to the transpose)
* We need to load blocks of Q horizontally and blocks of K vertically
* We do the same for `V` as `V` shows up on the right side of the GEMM computation: $softmax(QK^T) × V$.
    * The attention weights are applied to V
    * `V` needs to align with K on the sequence dimension

So `V` blocks are loaded in the same pattern as `K` blocks.

###### Compute dot product
Now that we've loaded blocks of Q, K and V in shared memory, we need to perform the core attention computation. i.e. we need to do:

$$
S = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V
$$

Let's break it down, let's first do: $QK^T$. We have the function `compute_dot_product` which is called as follows:
```cpp
// Each warp processes its assigned query rows
for (int local_row = 0; local_row < rows_per_warp; local_row++) {
    // Find absolute row index
    int row_idx = warp_row_offset + local_row;
    
    // Skip processing if the row is out of bounds
    if (row_idx >= indices.valid_rows) continue;
    
    // Compute attention scores for this row against all valid columns
    for (int col_idx = 0; col_idx < indices.valid_cols; col_idx++) {
        // Skip if using causal mask and col_idx is after row_idx
        int abs_row = indices.row_start + row_idx;
        int abs_col = indices.col_start + col_idx;
        if (ctx.is_causal && abs_col > abs_row) continue;
        
        // Calculate QK^T dot product
        float qk = compute_dot_product<T, D_VALUE>(
            s_Q[row_idx], s_K[col_idx], ctx.scale, warp
        );
```
What are `row_idx`, `col_idx` and `warp` variables? Let's inspect the for loop. The variable `local_row` is used to get the absolute row index: `row_idx`. `local_row`'s upper limit is `rows_per_warp` which is calculated as: 

```cpp
const int warp_count = blockDim.x / 32;
     
// Each warp handles specific rows
const int rows_per_warp = BLOCK_SIZE_M / warp_count;
```
We've set `BLOCK_SIZE_M` to `64` and `blockDim.x` is the number of threads per block which is set to `256`. So that makes `warp_count` = `8` and `rows_per_warp` = `8` as well. Each warp is responsible for **8** rows and we have a total of **8** warps per block.

Imagine we're in warp 0, `local_row` goes from `[0, 8)` in `compute_dot_product` we send `S_Q[0...7]` and each thread now is responsible for values at indices: `threadIdx + (n - 1) * 32`, where `n = D_VALUE / 32`, so thread 0 is responsible for indices 0, 32, 64 and 96.

We do the same thing for `s_K` shared memory region. 

Calculating `Q @ K.T` is similar to doing the dot product between rows of `Q` and `K`. Why is that the case? Well, the shape of `Q` is `B, H, N, D` and the shape of `K` is also `B, H, N, D`. For matrix multiplication, we need `K` to be transposed, but if we could access the rows of `K` efficiently (coalesced memory access), then in order to calculate one element of the attention matrix we need to multiply `D` values from `Q` and `D` values from `K`. This way, we’ll have an output of shape: `B, H, N, N`.

Each warp consists of 32 threads, and we unroll dot product computation across D_VALUE (128 in your case). Instead of each thread handling a single element, we distribute the workload such that:
* Thread i processes elements:

$$

i, i + 32, i + 64, i + 96

$$

* This ensures memory coalescing when fetching data from shared memory, as each thread accesses consecutive memory locations spaced 32 apart.


Let's now peek into `compute_dot_product`:

```cpp
const int lane_idx = threadIdx.x % 32;
T qk_sum = static_cast<T>(0);

// Each thread computes partial dot product
#pragma unroll
for (int d = lane_idx; d < D_VALUE; d += 32) {
    qk_sum += q_row[d] * k_row[d];
}

// Reduce within the warp
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    qk_sum += warp.shfl_down(qk_sum, offset);
}

// Apply scaling
if (lane_idx == 0) {
    qk_sum *= scale;
}

// Broadcast to all threads
return warp.shfl(qk_sum, 0);
```

`qk_sum` represents the result of the dot product of one row of `Q` with the corresponding row of `K`, we've written it such that each thread in a warp is responsible for multiplying more than one element. For `D_VALUE = 128`, thread `0` is responsible for indices: `0, 32, 64, 96`, thread 1 is responsible for `1, 33, 65 97` and so on. We reduce `qk_sum` using the standard reduction operation (we've implemented this on [Day 14](https://github.com/vectorquantized/100daysofcuda/blob/main/README.md#day-14-atomic-sum)) and multiply it with the `scale` value. 

After that we broadcast this value to all the threads for further processing. Only thread 0 in each warp holds the final dot product sum after reduction.
* To ensure all threads in the warp can use this result in the next stage, we broadcast it across the warp.