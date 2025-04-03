## CUTLASS notes
On day 66, I am starting with a goal to learn cutlass and understand how the abstractions in the library are laid out and how they help a CUDA developer to write really awesome kernels. As NVIDIA documentation mentions [here](https://github.com/NVIDIA/cutlass/blob/main/README.md):

> CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.  It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS and cuDNN. CUTLASS decomposes these "moving parts" into reusable, modular software components abstracted by C++ template classes. Primitives for different levels of a conceptual parallelization hierarchy can be specialized and tuned via custom tiling sizes, data types, and other algorithmic policy. The resulting flexibility simplifies their use as building blocks within custom kernels and applications.

It also provides 
* Support for mixed-precision computations, 
* Specialized data-movement 
* Multiply-accumulate abstractions for FP64, FP32, TF32, FP16, BF16, [FP32 emulation via tensor core instruction](https://github.com/NVIDIA/cutlass/blob/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm)

> CUTLASS implements high-performance convolution via the implicit GEMM algorithm. Implicit GEMM is the formulation of a convolution operation as a GEMM thereby taking advantage of CUTLASS's modular GEMM pipeline. This allows CUTLASS to build convolutions by reusing highly-optimized GEMM components.


### Getting started

#### Setup
We start with a set of pre-requisites mentioned on their [quickstart guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/quickstart.md#prerequisites)

I have done the following already for days 64 and 65:

```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=86 # as I am on a runpod pod, using a RTX A5000
```
Device properties of the device running on the pod I am on:
```python
import torch
torch.cuda.get_device_properties(torch.cuda.current_device())
```
output: 
```
_CudaDeviceProperties(name='NVIDIA RTX A5000', major=8, minor=6, total_memory=24138MB, multi_processor_count=64, uuid=dc1345df-ac63-34db-00eb-1bddd0042c46, L2_cache_size=6MB)
```

#### Toy exercise
So, we start by performing a toy exercise just like the documentation:

```cpp
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main(int argc, char* argv[]) {
    cutlass::half_t x = 2.25_hf;
    std::cout << "x: " << x << std::endl;
    return 0;
}
```

Since we've created a CMakelists.txt and everything is configured, all we need to do is, just step out of the day_66 folder and reach build folder and build & run.

```bash
cd ../../
mkdir -p build & cd build
cmake ..
make day_66
./day_66
```
We get the following output:
```
x: 2.25
```

Phew! types work! :joy:. Moving on to what I am here for: GEMM!

#### GEMM 


