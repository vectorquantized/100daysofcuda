### Day 28 Batched convolution with input channels

#### Setup
Input is of shape: `B, C, H, W`, kernel is of shape: `1, C, k_h, h_w`, we are performing valid convolution as before.

#### Implementation
Previously we weren't able to match the reference implementation. The reference implementation is given by:
```python
B, C_in, H, W = 32, 3, 1024, 1024
kernel_size = (3, 3)

weight = torch.randn(C_in, *kernel_size, device="cuda", dtype=torch.float32)
input = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
output = torch.zeros((B, H - kernel_size[0] + 1, W - kernel_size[1] + 1), device=input.device, dtype=input.dtype)

C_ref = F.conv2d(input, weight[None, ...]).squeeze(1)
```
* Since we're doing a tiled implementation our strategy for the kernel is as follows:
    * Process each input channel
    * Load one channel, calculate the result
    * Accumulate the subsequent convs on accumulator value until all channels are done.
* Boundary conditions play a critical role to get the correct output.
    * Current thread that is producing the output, should stay within bounds.
    * We want the `row_out` and `col_out` to be smaller than output size, given by `out_h` and `out_w`
    * Since each thread is responsible for calculating the output of each cell, we make each thread calculate the output for one output cell by iterating over kernel values. This means that the nested for loop over kernel values is iterated upon by each output thread.

#### Results
When the aforementioned strategy is followed, we get correct results. We also observe that the kernel is faster than `F.conv2D`, which is very rewarding!

```
python profile_conv2d.py 
input.shape=torch.Size([32, 3, 1024, 1024]), weight.shape=torch.Size([3, 3, 3])
C_custom.shape=torch.Size([32, 1022, 1022])
C_ref.shape=torch.Size([32, 1022, 1022])
Results match: True
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  Total GFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           torch_conv2d         7.46%     415.652us        17.16%     955.994us     955.994us       0.000us         0.00%       2.135ms       2.135ms             1            --  
                                           aten::conv2d         0.13%       7.483us         8.75%     487.382us     487.382us       0.000us         0.00%       2.135ms       2.135ms             1         1.805  
                                      aten::convolution         0.21%      11.570us         8.61%     479.899us     479.899us       0.000us         0.00%       2.135ms       2.135ms             1            --  
                                     aten::_convolution         3.24%     180.548us         8.41%     468.329us     468.329us       0.000us         0.00%       2.135ms       2.135ms             1            --  
                                aten::cudnn_convolution         1.12%      62.315us         5.17%     287.781us     287.781us       2.135ms        55.27%       2.135ms       2.135ms             1            --  
void cudnn::cnn::conv2d_grouped_direct_kernel<false,...         0.00%       0.000us         0.00%       0.000us       0.000us       2.135ms        55.27%       2.135ms       2.135ms             1            --  
                                           torch_conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       2.135ms        55.27%       2.135ms       2.135ms             1            --  
void conv2d_tiled_channel<float>(float const*, float...         0.00%       0.000us         0.00%       0.000us       0.000us       1.728ms        44.73%       1.728ms       1.728ms             1            --  
                                          custom_conv2d         0.00%       0.000us         0.00%       0.000us       0.000us       1.728ms        44.73%       1.728ms       1.728ms             1            --  
                                          custom_conv2d        40.05%       2.231ms        40.58%       2.261ms       2.261ms       0.000us         0.00%       0.000us       0.000us             1            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 5.571ms
Self CUDA time total: 3.863ms

Max diff = 0.0

```
`torch.conv2d` takes 2.135 ms, while the custom kernel takes 1.728 ms, the difference could also be that we don't allocate any memory inside our custom implementation. We expect the user to send the allocated output storage to the kernel. The memory allocation and transfer time could be seen in custom kernel's CPU time.

