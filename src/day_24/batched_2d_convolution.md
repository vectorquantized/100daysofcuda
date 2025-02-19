### Batched 2D convlution
On day 17 we tried to write up the batched 2d convolution kernel. There needs to be extra care taken to account for output channels and input_channels.

Weights are initialized as:

```python
B, C_in, C_out, H, W = 16, 3, 64, 128, 128
kernel_size = (3, 3)

weight = torch.randn(C_out, C_in, *kernel_size, device="cuda", dtype=torch.float32)
```

Keeping the above dimensions in mind, we need to first allocate the output tensor in the custom kernel as:
```cpp
auto output = torch::zeros({batch_size, out_channels, out_rows, out_cols}, 
        torch::TensorOptions().device(input.device()).dtype(input.dtype()));
```

We also need to pay special attention to the indexing of he kernel and the output inside the kernel.

```cpp
for (int out_c = 0; out_c < out_channels; ++out_c) {
    T p_value = static_cast<T>(0);
    for (int in_c = 0; in_c < in_channels; in_c++) {
        const T* input_channel = input + batch_idx * in_channels * width * height + in_c * width * height;
        const T* mask_channel = conv_mask + (out_c * in_channels + in_c) * filter_size * filter_size;
```

Notice that for mask_channel we add (out_c * in_channels + in_c) * filter_size * filter_size
