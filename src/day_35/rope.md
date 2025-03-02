## Rotary Positional Embedding
Rotary Position Embedding (RoPE) is a relative positional encoding method used in training almost all of the modern LLMs based on Transformer Decoder block. The idea is to encode the relative position information directly into the attention mechanism by applying rotations in a complex vector space. one benefit of RoPE being relative and not absolute is that we can do context length extension during post training. There are no extra training parameters, just extra few matmuls are needed which can be fused with the attention mechanism.

We split the input into even and odd pairs and apply rotation matrix.

$$
q = [q_0, q_1, q_2, q_3, …, q_{D-2}, q_{D-1}]
$$

where each pair $(q_i, q_{i+1})$ represents a 2D coordinate. Each pairr is rotated by an angle theta that depends on the position as:

$$
\theta_d = \frac{10000^{-2d/D}}{10000}
$$

The rotation is applied using the 2D rotation matrix as follows:

$$

\begin{bmatrix} q_i{\prime} \\ q_{i+1}{\prime} \end{bmatrix}

\begin{bmatrix} \cos(\theta_p) & -\sin(\theta_p) \\ \sin(\theta_p) & \cos(\theta_p) \end{bmatrix}
\begin{bmatrix} q_i \\ q_{i+1} \end{bmatrix}

$$

In code we do something like the following:
```python
def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies Rotary Position Embeddings (RoPE) to the input tensor.

    Args:
        q (torch.Tensor): Query tensor of shape (B, L, H, D).
        k (torch.Tensor): Key tensor of shape (B, L, H, D).
        cos (torch.Tensor): real part of the embs, shape: (L, D/2)
        sin (torch.Tensor): imaginary part of the embs, shape: (L, D/2)

    Returns:
        torch.Tensor: The tensor after applying RoPE.
    """
    B, L, H, D = q.shape
    assert D % 2 == 0, "Head dimension must be even for RoPE."
    assert q.shape == k.shape, f"Query and key shapes should match, got: {q.shape=} and {k.shape=}"

    # Expand for broadcasting
    sin = sin.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)
    cos = cos.unsqueeze(0).unsqueeze(2)  # Shape: (1, L, 1, D/2)
    q_rot = rotate(q, cos, sin)
    k_rot = rotate(k, cos, sin)
    
    return q_rot, k_rot
```

The `rotate` function is just splitting the input into even and odd halves and doing the multiplication:

```python
# Select even and odd indices
x_even = x[..., ::2]  # Shape: (B, L, H, D/2)
x_odd = x[..., 1::2]  # Shape: (B, L, H, D/2)

# Apply rotation
x_rot = torch.empty_like(x)
x_rot[..., ::2] = x_even * cos - x_odd * sin
x_rot[..., 1::2] = x_odd * cos + x_even * sin
```

We implement the same operation in a CUDA kernel. Our kernel configuration looks something like the following:

```cpp
dim3 threads(std::min(128, (D + 1) / 2));
dim3 blocks(
    (D / 2 + threads.x - 1) / threads.x,
    L,
    B * H
);
```
Each block has a maximum of 128 threads and each grid is laid out as `(D/2, L, B * H)`. We calculate the base and position indices as follows:
```cpp
int dim_offset  = threadIdx.x + blockIdx.x * blockDim.x;
int base_idx =((batch_idx * ctx.seq_len  + seq_idx) * ctx.num_heads + head_idx) * ctx.head_dim;
int pos_id = ctx.position_ids[batch_idx * ctx.seq_len + seq_idx];
```

Let's do some basic caluations here with the following setup:
```
threadIdx.x = 12 // (can't be more than 16 as we launch with min(128, 32 / 2))
blockIdx.x = 1
blockDim.x = 16 // D / 2
dim_offset = 12 + 1 * 16 = 28
batch_idx = 1 // second batch
seq_idx = 10 // 10th position in the sequence
seq_len = 128 // sequence length
num_heads = 4 // number of heads
head_idx = 0 // index of the head
head_dim = 32 // dimension of each head

base_idx = 1 * 128 * 4 * 32 + 10 * 4 * 32 + 0 * 32 = 138 * 128 = 17,664
dim_idx = base_idx + dim_offset * 2 = base_idx + 56

pos_idx = ctx.position_ids[1 * 128 + 10 = 138]
```

The current thread is responsible for calculating the embeddings for 28th pair of coordinates. `dim_idx = base_idx + 56`. We reach `cos_pos` by adding `pos_idx * D / 2` to `cos` base address (as each position has `D/2` values for `cos` and `D/2` values for `sin`)

We then read the values of `Q` and `K` using `dim_idx` and `dim_idx + 1` and rotate them using:
```
out_even = even * cos - odd * sin
out_odd = odd * cos + even * sin
```

We profile the code by comparing it to the PyTorch reference implementation, and observe the following results:
```
python profile_rope.py
Custom CUDA implementation: 3.55 us
PyTorch implementation: 208.88 us
Speedup: 58.80x
Maximum query difference: 4.76837158203125e-07
Maximum key difference: 4.76837158203125e-07
Results match: True
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  Total MFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             torch_rope         0.00%       0.000us         0.00%       0.000us       0.000us      73.555ms        82.97%      73.555ms      73.555ms             1            --  
                                             torch_rope         0.36%     321.677us         0.75%     679.623us     679.623us       0.000us         0.00%      73.544ms      73.544ms             1            --  
                                              aten::mul         0.09%      79.572us         0.15%     130.805us      16.351us      36.031ms        40.64%      36.031ms       4.504ms             8      2147.484  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      36.031ms        40.64%      36.031ms       4.504ms             8            --  
                                            aten::copy_         0.04%      35.691us         0.06%      55.552us      13.888us      19.768ms        22.30%      19.768ms       4.942ms             4            --  
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us      19.768ms        22.30%      19.768ms       4.942ms             4            --  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      17.745ms        20.02%      17.745ms       4.436ms             4            --  
void apply_rope<float, int>(float const*, float cons...         0.00%       0.000us         0.00%       0.000us       0.000us      15.113ms        17.05%      15.113ms      15.113ms             1            --  
                                            custom_rope         0.00%       0.000us         0.00%       0.000us       0.000us      15.113ms        17.05%      15.113ms      15.113ms             1            --  
                                              aten::sub         0.02%      19.654us         0.03%      29.402us      14.701us       8.880ms        10.02%       8.880ms       4.440ms             2            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 90.154ms
Self CUDA time total: 88.657ms
```

We see that the speed up is ~58x, the reason for this is because the PyTorch implementation is using pure matmuls, while the custom cuda kernel is able to operate on even and odd elements in one go.

```python
x_rot[..., ::2] = x_even * cos - x_odd * sin
x_rot[..., 1::2] = x_odd * cos + x_even * sin
```
is costlier to perform than inside one kernel. So, there's implicit fusion of operations built into the CUDA kernel. We could further optimize the kernel by optimizing for occupancy and register pressure and also utilize shared memory but we'll explore that possibility in the next kernel implementation where we can just fuse the RoPE op with attention implementation perhaps.
