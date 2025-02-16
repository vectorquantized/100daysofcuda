### LLM FeedForward Layer
On day 20 we tried to write up the LLM FeedFoward layer. The FF layer looks like the following:

```python
torch_feed_forward=SwigLU(
  (gate_proj): Linear(in_features=512, out_features=1792, bias=False)
  (up_proj): Linear(in_features=512, out_features=1792, bias=False)
  (down_proj): Linear(in_features=1792, out_features=512, bias=False)
  (activation_fn): SiLU()
)
```

The forward function will do the following:
```python
def forward( # pylint: disable=arguments-differ
         self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through feedForward Layer.

        Args:
            input_batch: a torch tensor of shape: (B, L, D)

        Returns:
            a torch tensor of shape: (B, L, D)
        """
        gated = self.gate_proj(input_batch)
        x = self.up_proj(input_batch)  # B, L, hidden_dim
        x = x * self.activation_fn(gated)  # B, L, hidden_dim
        x = self.down_proj(x)  # B, L, D
        return x
```

Kernel Strategy:
* `gate_proj(input_batch) and up_proj(input_batch)` are matmuls
* In ` x * self.activation_fn(gated)` , `activation_fn` is `SiLU`.
* `down_proj(x)` is another matmul.

The naive way to implement this kernel is to have 4 different kernel invocations and that leaves plenty of opportunities to optimize the kernel by virtue of fusion.

Below is the comparison we run against PyTorch. We use our own tiled gemm implementation.

```
Matrices match: True
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                    custom_feed_forward         0.00%       0.000us         0.00%       0.000us       0.000us      10.877ms        90.14%      10.877ms      10.877ms           0 b           0 b           0 b           0 b             1            --  
                                     torch_feed_forward         0.00%       0.000us         0.00%       0.000us       0.000us      10.502ms        87.04%      10.502ms      10.502ms           0 b           0 b           0 b           0 b             1            --  
void bmm_broadcast_B<float, 16, float>(float const*,...         0.00%       0.000us         0.00%       0.000us       0.000us      10.114ms        83.82%      10.114ms       3.371ms           0 b           0 b           0 b           0 b             3            --  
                                     torch_feed_forward         0.75%     686.944us        82.39%      75.465ms      75.465ms       0.000us         0.00%       1.685ms       1.685ms           0 b           0 b      16.12 Mb    -112.00 Mb             1            --  
                                           aten::linear         0.05%      47.257us        71.13%      65.155ms      21.718ms       0.000us         0.00%       1.466ms     488.692us           0 b           0 b      72.12 Mb           0 b             3            --  
                                           aten::matmul         0.14%     126.722us        71.02%      65.055ms      21.685ms       0.000us         0.00%       1.466ms     488.692us           0 b           0 b      72.12 Mb           0 b             3            --  
                                               aten::mm        53.46%      48.972ms        70.85%      64.896ms      21.632ms       1.466ms        12.15%       1.466ms     488.692us           0 b           0 b      72.12 Mb      72.12 Mb             3     22548.578  
                                 ampere_sgemm_128x64_tn         0.00%       0.000us         0.00%       0.000us       0.000us       1.466ms        12.15%       1.466ms     488.511us           0 b           0 b           0 b           0 b             3            --  
                                    custom_feed_forward         0.41%     376.891us        17.61%      16.131ms      16.131ms       0.000us         0.00%     630.943us     630.943us           0 b           0 b       8.00 Mb     -84.00 Mb             1            --  
                                  cudaDeviceSynchronize        11.64%      10.664ms        11.64%      10.664ms       1.777ms     494.623us         4.10%     494.623us      82.437us           0 b           0 b           0 b           0 b             6            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 91.597ms
Self CUDA time total: 12.066ms

```