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

So, following the example for **Turing** Architecture in the Nvidia tutorial was easier but then I am running on **Ampere** and just changing the compute architecture to `cutlass::arch::Sm80` wouldn't work, which was puzzling. So, I tried a few things, and then eventually found out that for **Ampere** one has to explicitly pass in the following configurations:
```cpp
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    float,
    float
>
```

Which made me read the documentation a bit more, so, [here's](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm.h#L233) what the "signature" of `GEMM` class looks like:
```cpp
using ElementA = ElementA_;
using LayoutA = LayoutA_;
using TensorRefA = TensorRef<ElementA const, LayoutA>;
using ElementB = ElementB_;
using LayoutB = LayoutB_;
using TensorRefB = TensorRef<ElementB const, LayoutB>;
using ElementC = ElementC_;
using LayoutC = LayoutC_;
using TensorRefC = TensorRef<ElementC const, LayoutC>;
using TensorRefD = TensorRef<ElementC, LayoutC>;
using ElementAccumulator = ElementAccumulator_;
using OperatorClass = OperatorClass_;
using ArchTag = ArchTag_;
using ThreadblockShape = ThreadblockShape_;
using WarpShape = WarpShape_;
using InstructionShape = InstructionShape_;
using EpilogueOutputOp = EpilogueOutputOp_;
using ThreadblockSwizzle = ThreadblockSwizzle_;
using Operator = Operator_;
static int const kStages = Stages;
static int const kAlignmentA = AlignmentA;
static int const kAlignmentB = AlignmentB;
static int const kAlignmentC = EpilogueOutputOp::kCount;
static bool const kSplitKSerial = SplitKSerial;
static ComplexTransform const kTransformA = ComplexTransform::kNone;
static ComplexTransform const kTransformB = ComplexTransform::kNone;
```

Keeping this in mind and extending my code to be able to run on both Turing and Ampere architectures, I did the following:
```cpp
template<typename ArchTag_>
struct GemmConfig {
    using ArchTag = ArchTag_;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float
    >;

};
```

The above code creates a template for `GemmConfig` and given an `ArchTag_` we can dispatch the correct GEMM call. The default Gemm configuration used bvy cutlass is:
```cpp
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,  ///< Number of elements computed per operation.
    ElementAccumulator,
    ElementAccumulator
  >;
  using OperatorClass = cutlass::arch::OpClassSimt,

```
So, the `InstructionShape` and the `Count` parameter of `LinearConbination` are different for **Ampere** and that is why we'd get a GEMM ERROR if we simply replaced the ArchTag from `Sm75` to `Sm80`. Having solved this mystery, I launched the kernel as given in the .cu file in `day_67` and voila! Things work magically!
One observation though is that [`DefalutGemmConfiguration`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/default_gemm_configuration.h#L73) values don't work for `Sm75`. So the following config will not work:
```cpp
template<>
struct GemmConfig<cutlass::arch::Sm75> {
    // Need to add this
    using ArchTag = cutlass::arch::Sm75;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>; 
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    // For SIMT operations instead of tensor ops
    using OperatorClass = cutlass::arch::OpClassSimt;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,  // Vector width of 1
        ElementAccumulator,
        ElementAccumulator
    >;
};
```