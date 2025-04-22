#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/array.h>
#include "../../cuda/cutlass/util.h"
#include "../../cuda/softmax.h"
#include <vector>

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

template<typename ArchTag_>
struct TiledGemmConfig {
    using ArchTag = ArchTag_;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float
    >;
};

template<
    typename ArchTag_,
    typename EpilogueOp_= cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    float,
    float
    >
>
struct GemmConfigWithEpilogue {
    using ArchTag = ArchTag_;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using EpilogueOp = EpilogueOp_;
    static int const NumStages = 2;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
};

template<typename Element, typename Layout>
class InitPolicy {
public:
    virtual ~InitPolicy() = default;
    virtual void initialize(cutlass::TensorRef<Element, Layout> ref, int rows, int cols) const = 0;

};

template<typename Element, typename Layout>
class GaussianInitPolicy : public InitPolicy<Element, Layout> {
private:
    uint64_t seed;
    Element mean;
    Element stddev;
    int bits_less_than_one;

public:
    GaussianInitPolicy(uint64_t seed_, Element mean_, Element stddev_, int bits_less_than_one_)
    : seed(seed_), mean(mean_), stddev(stddev_), bits_less_than_one(bits_less_than_one_)
    {}

    void initialize(cutlass::TensorRef<Element, Layout> tensor_ref, int rows, int cols) const {
        auto tensor_view = cutlass::TensorView<Element, Layout>(
            tensor_ref.data(), tensor_ref.layout(), {rows, cols}
        );
        cutlass::reference::device::TensorFillRandomGaussian(
            tensor_view,
            seed,
            mean,
            stddev,
            bits_less_than_one
        );
    }
};

template<typename Element, typename Layout>
class UniformInitPolicy : public InitPolicy<Element, Layout> {
private:
    uint64_t seed;
    Element extent_low;
    Element extent_high;
    int bits_less_than_one;

public:
    UniformInitPolicy(uint64_t seed_, Element extent_low_, Element extent_high_, int bits_less_than_one_)
    : seed(seed_), extent_low(extent_low_), extent_high(extent_high_), bits_less_than_one(bits_less_than_one_)
    {}

    void initialize(cutlass::TensorRef<Element, Layout> tensor_ref, int rows, int cols) const {
        auto tensor_view = cutlass::TensorView<Element, Layout>(
            tensor_ref.data(), tensor_ref.layout(), {rows, cols}
        );
        cutlass::reference::device::TensorFillRandomUniform(
            tensor_view,
            seed,
            extent_low,
            extent_high,
            bits_less_than_one
        );
    }
};

template<typename Element, typename Layout>
class UniformInitPolicyKernel : public InitPolicy<Element, Layout> {
private:
    uint64_t seed;
    Element extent_low;
    Element extent_high;
    int bits_less_than_one;

public:
    UniformInitPolicyKernel(uint64_t seed_, Element extent_low_, Element extent_high_, int bits_less_than_one_)
    : seed(seed_), extent_low(extent_low_), extent_high(extent_high_), bits_less_than_one(bits_less_than_one_)
    {}

    void initialize(cutlass::TensorRef<Element, Layout> tensor_ref, int rows, int cols) const {
        auto tensor_view = cutlass::TensorView<Element, Layout>(
            tensor_ref.data(), tensor_ref.layout(), {rows, cols}
        );

        int total_elements = rows * cols;
        dim3 grid((total_elements + 255) / 256);
        dim3 block(256);

        uniform_random_fill<cutlass::half_t><<<grid, block>>>(
            tensor_ref.data(),
            total_elements,
            seed,
            extent_low,
            extent_high,
            bits_less_than_one
        );
    }
};

template<typename Element, typename Layout>
struct GaussianInitializer {
    uint64_t seed;
    Element mean;
    Element stddev;
    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one;

    GaussianInitializer(uint64_t seed_, Element mean_, Element stddev_, int bits_less_than_one_)
    : seed(seed_), mean(mean_), stddev(stddev_), bits_less_than_one(bits_less_than_one_) {}

    void operator()(cutlass::TensorRef<Element, Layout> tensor_ref, int rows, int cols) const {
        auto tensor_view = cutlass::TensorView<Element, Layout>(
            tensor_ref.data(), tensor_ref.layout(), {rows, cols}
        );

        cutlass::reference::device::TensorFillRandomGaussian(
            tensor_view,
            seed,
            mean,
            stddev,
            bits_less_than_one
        );
    }
};

//GEMM for a given architecture config.
template <typename Config>
cutlass::Status run_gemm(int M, int N, int K, float alpha, float beta) {

    using Gemm = cutlass::gemm::device::Gemm<
        typename Config::ElementA,
        typename Config::LayoutA,
        typename Config::ElementB,
        typename Config::LayoutB,
        typename Config::ElementC,
        typename Config::LayoutC,
        typename Config::ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        typename Config::ArchTag,
        typename Config::ThreadblockShape,
        typename Config::WarpShape,
        typename Config::InstructionShape,
        typename Config::EpilogueOp
    >;

    Gemm gemm_op;

    // Allocate device memory via CUTLASS host tensors.
    cutlass::HostTensor<typename Config::ElementA, typename Config::LayoutA> A({M, K});
    cutlass::HostTensor<typename Config::ElementB, typename Config::LayoutB> B({K, N});
    cutlass::HostTensor<typename Config::ElementC, typename Config::LayoutC> C({M, N});

    // Launch GEMM
    cutlass::Status status = gemm_op({
        {M, N, K},    // Problem size
        A.device_ref(),  // TensorRef for A
        B.device_ref(),  // TensorRef for B
        C.device_ref(),  // TensorRef for C (input/output)
        C.device_ref(),  // TensorRef for D (output)
        {alpha, beta}    // Epilogue arguments
    });

    return status;
}

template<
    typename Config, 
    typename InitializerA = std::nullptr_t,
    typename InitializerB = std::nullptr_t,
    typename InitializerC = std::nullptr_t
>
cutlass::Status run_gemm(
    int M, int N, int K, 
    float alpha, float beta,
    InitializerA init_A = nullptr,
    InitializerB init_B = nullptr,
    InitializerC init_C = nullptr
) {
    using Gemm = cutlass::gemm::device::Gemm<
    typename Config::ElementA,
    typename Config::LayoutA,
    typename Config::ElementB,
    typename Config::LayoutB,
    typename Config::ElementC,
    typename Config::LayoutC,
    typename Config::ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    typename Config::ArchTag,
    typename Config::ThreadblockShape,
    typename Config::WarpShape,
    typename Config::InstructionShape,
    typename Config::EpilogueOp
    >;

    Gemm gemm_op;
    // Allocate device memory via CUTLASS host tensors.
    cutlass::HostTensor<typename Config::ElementA, typename Config::LayoutA> A({M, K});
    cutlass::HostTensor<typename Config::ElementB, typename Config::LayoutB> B({K, N});
    cutlass::HostTensor<typename Config::ElementC, typename Config::LayoutC> C({M, N});
    if constexpr(!std::is_same_v<InitializerA, std::nullptr_t>) {
        init_A(A.device_ref(), M, K);
    }
    if constexpr(!std::is_same_v<InitializerB, std::nullptr_t>) {
       init_B(B.device_ref(), K, N); 
    }
    if constexpr(!std::is_same_v<InitializerC, std::nullptr_t>) {
        init_C(C.device_ref(), M, N);
    }
    cutlass::Status status = gemm_op({
        {M, N, K},
        A.device_ref(),
        B.device_ref(),
        C.device_ref(),
        C.device_ref(),
        {alpha, beta}
    });
    return status;
}

/* */

template<typename Element>
cutlass::Status run_gemm_batched_array(
    int M, int N, int K, Element alpha, 
    Element const* const* A, int lda, 
    Element const* const* B, int ldb, 
    Element* const* C, int ldc,
    Element beta,
    int batch_count) {
    
    using Gemm = cutlass::gemm::device::GemmArray<
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor
        >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {M, N, K},
        A, lda,
        B, ldb,
        C, ldc,
        C, ldc,
        {alpha, beta},
        batch_count
    });
    return status;
}

template<typename Element>
cutlass::Status run_gemm_batched(
    int M, int N, int K, Element alpha, 
    Element const* A, int lda,
    int batch_stride_A,
    Element const* B, int ldb, 
    int batch_stride_B,
    Element* C, int ldc,
    int batch_stride_C,
    Element beta,
    int batch_count) {
    
    using Gemm = cutlass::gemm::device::GemmBatched<
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor
        >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {M, N, K},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });
    return status;
}

template<typename Element>
cutlass::Status attention_scores(
    int M, int N, int K, Element alpha, 
    Element const* A, int lda,
    int batch_stride_A,
    Element const* B, int ldb, 
    int batch_stride_B,
    Element* C, int ldc,
    int batch_stride_C,
    Element beta,
    int batch_count) {
    
    using Gemm = cutlass::gemm::device::GemmBatched<
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::ColumnMajor
        >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {M, N, K},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    float epsilon = 1e-5f;  // Small value for numerical stability

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int num_rows = batch_count * M;  // Total number of rows across all batches
    int blocks_needed = (num_rows + threads_per_block - 1) / threads_per_block;

    dim3 grid(blocks_needed);
    dim3 block(threads_per_block);

    // Launch the softmax kernel
    batched_online_softmax<float><<<grid, block>>>(
        C,   // Input from GEMM (Q*K^T)
        C,   // Output (same as input for in-place operation)
        batch_count,       // Number of batches
        M,                 // Sequence length (rows per batch)
        N,                 // Dimension to normalize over (columns per row)
        epsilon            // Small value for numerical stability
    );
    return status;
}

template<typename Element>
cutlass::Status attention(
    int M, int N, int K, Element alpha, 
    Element const* A, int lda,
    int batch_stride_A,
    Element const* B, int ldb, 
    int batch_stride_B,
    Element* C, int ldc,
    int batch_stride_C,
    Element const* D, int ldd,
    int batch_stride_D,
    Element beta,
    int batch_count) {
    
    using Gemm = cutlass::gemm::device::GemmBatched<
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::ColumnMajor
        >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {M, N, K},
        {A, lda},
        batch_stride_A,
        {B, ldb},
        batch_stride_B,
        {C, ldc},
        batch_stride_C,
        {C, ldc},
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

     // Synchronize and check errors after softmax kernel launch.
     cudaError_t cudaStatus = cudaDeviceSynchronize();
     if (cudaStatus != cudaSuccess) {
         std::cerr << "Softmax kernel error: " << cudaGetErrorString(cudaStatus) << std::endl;
         return cutlass::Status::kErrorInternal;
     }

    float epsilon = 1e-5f;  // Small value for numerical stability

    // Configure kernel launch parameters
    int threads_per_block = 256;
    int num_rows = batch_count * M;  // Total number of rows across all batches
    int blocks_needed = (num_rows + threads_per_block - 1) / threads_per_block;

    dim3 grid(blocks_needed);
    dim3 block(threads_per_block);

    // Launch the softmax kernel
    batched_online_softmax<float><<<grid, block>>>(
        C,   // Input from GEMM (Q*K^T)
        C,   // Output (same as input for in-place operation)
        batch_count,       // Number of batches
        M,                 // Sequence length (rows per batch)
        N,                 // Dimension to normalize over (columns per row)
        epsilon            // Small value for numerical stability
    );

    using Gemm_attn = cutlass::gemm::device::GemmBatched<
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor,
        Element, cutlass::layout::ColumnMajor
        >;

    Gemm_attn gemm_attn_op;

    // C: B, M, N and V: B, M, K.
    // output: B, M, K
    // problem size should be: M, K, N
    status = gemm_attn_op({
        {M, K, N},
        {C, ldc}, // Left-hand operand: softmax scores
        batch_stride_C,
        {D, ldd}, // Right-hand operand: V matrix
        batch_stride_D,
        {C, ldc}, // Output: in-place in C (or use another tensor if desired)
        batch_stride_C,
        {C, ldc}, // Accumulator
        batch_stride_C,
        {alpha, beta},
        batch_count
    });

    return status;
}

template<typename Element, typename Config>
cutlass::Status run_gemm_split_k(
    int M, int N, int K, 
    Element alpha, Element beta,
    int split_k_slices,
    const InitPolicy<typename Config::ElementA, typename Config::LayoutA>* init_policy_A = nullptr,
    const InitPolicy<typename Config::ElementB, typename Config::LayoutB>* init_policy_B = nullptr,
    const InitPolicy<typename Config::ElementC, typename Config::LayoutC>* init_policy_C = nullptr) {
    
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    using Gemm = cutlass::gemm::device::GemmSplitKParallel<
        typename Config::ElementA,
        typename Config::LayoutA,
        typename Config::ElementB,
        typename Config::LayoutB,
        typename Config::ElementC,
        typename Config::LayoutC,
        typename Config::ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        typename Config::ArchTag,
        typename Config::ThreadblockShape,
        typename Config::WarpShape,
        typename Config::InstructionShape,
        typename Config::EpilogueOp
        >;

        cutlass::Status status = cutlass::Status::kInvalid;
        cutlass::HostTensor<typename Config::ElementA, typename Config::LayoutA> A({M, K});
        cutlass::HostTensor<typename Config::ElementB, typename Config::LayoutB> B({K, N});
        cutlass::HostTensor<typename Config::ElementC, typename Config::LayoutC> C({M, N});
        if (init_policy_A != nullptr && init_policy_B != nullptr && init_policy_C != nullptr) {
            init_policy_A->initialize(A.device_ref(), M, K);
            init_policy_B->initialize(B.device_ref(), K, N);
            init_policy_C->initialize(C.device_ref(), M, N);

            typename Gemm::Arguments arguments{
                problem_size,
                A.device_ref(),
                B.device_ref(),
                C.device_ref(),
                C.device_ref(),
                {alpha, beta},
                split_k_slices};

            size_t workspace_size = Gemm::get_workspace_size(arguments);
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

            Gemm gemm_op;
            status = gemm_op.initialize(arguments, workspace.get());
        }
        return status;
}

template<typename Element, typename Config>
cutlass::Status run_gemm_with_activation(
    int M, int N, int K, 
    Element alpha, Element beta,
    const InitPolicy<typename Config::ElementA, typename Config::LayoutA>* init_policy_A = nullptr,
    const InitPolicy<typename Config::ElementB, typename Config::LayoutB>* init_policy_B = nullptr,
    const InitPolicy<typename Config::ElementC, typename Config::LayoutC>* init_policy_C = nullptr
) {

    
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    using Gemm = cutlass::gemm::device::Gemm<
        typename Config::ElementA,
        typename Config::LayoutA,
        typename Config::ElementB,
        typename Config::LayoutB,
        typename Config::ElementC,
        typename Config::LayoutC,
        typename Config::ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        typename Config::ArchTag,
        typename Config::ThreadblockShape,
        typename Config::WarpShape,
        typename Config::InstructionShape,
        typename Config::EpilogueOp,
        typename Config::SwizzleThreadBlock,
        Config::NumStages
        >;

        cutlass::Status status = cutlass::Status::kInvalid;
        cutlass::HostTensor<typename Config::ElementA, typename Config::LayoutA> A({M, K});
        cutlass::HostTensor<typename Config::ElementB, typename Config::LayoutB> B({K, N});
        cutlass::HostTensor<typename Config::ElementC, typename Config::LayoutC> C({M, N});
        if (init_policy_A != nullptr && init_policy_B != nullptr && init_policy_C != nullptr) {
            init_policy_A->initialize(A.device_ref(), M, K);
            init_policy_B->initialize(B.device_ref(), K, N);
            init_policy_C->initialize(C.device_ref(), M, N);

            typename Gemm::Arguments arguments{
                problem_size,
                A.device_ref(),
                B.device_ref(),
                C.device_ref(),
                C.device_ref(),
                {alpha, beta}};

            size_t workspace_size = Gemm::get_workspace_size(arguments);
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

            Gemm gemm_op;
            gemm_op.initialize(arguments, workspace.get());
            status = gemm_op();                
            
        }
        return status;
}

template<
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_,
    typename ElementCompute_
>
class LinearCombinationSwish {
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementOutput = ElementOutput_;
    using ElementVector = cutlass::Array<ElementOutput, Count>;

    static int const kCount = Count;
    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
    };
private:
    Params params_;

public:
    CUTLASS_HOST_DEVICE LinearCombinationSwish(Params const &params) : params_(params) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const {
        return params_.beta != ElementCompute(0);
    }
    CUTLASS_DEVICE void operator() (
        ElementVector &d,
        ElementAccumulator const &accumulator,
        ElementVector const &source,
        ElementCompute alpha_scalar = ElementCompute(1),
        ElementCompute beta_scalar = ElementCompute(1)) const {
        
        CUTLASS_PRAGMA_UNROLL
        for(int i=0; i<Count; ++i) {
            ElementCompute compute(accumulator);
            compute = compute * params_.alpha * alpha_scalar;
            if (params_.beta != ElementCompute(0)) {
                    ElementCompute compute_source(source[i]);
                    compute = compute + params_.beta * beta_scalar * compute_source;
                }
            compute = apply_swish(compute);
            d[i] = ElementOutput(compute);
        }
        
    }

private:
    CUTLASS_DEVICE ElementCompute apply_swish(ElementCompute x) const {
        return x / (ElementCompute(1) + cutlass::fast_exp(-x));
    }
};


template<
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_,
    typename ElementCompute_
>
class LinearCombinationSwiglu {
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementOutput = ElementOutput_;
    using ElementVector = cutlass::Array<ElementOutput, Count>;

    static int const kCount = Count;
    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementOutput const* up_ptr;
        int ldm;
        int batch_stride;
    };
private:
    Params params_;

public:
    CUTLASS_HOST_DEVICE LinearCombinationSwiglu(Params const &params) : params_(params) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const {
        return params_.beta != ElementCompute(0);
    }
    // CUTLASS_HOST_DEVICE
    // void set_k_partition(int k_partition, int partitions_k) {}
    CUTLASS_DEVICE ElementVector operator()(
        ElementAccumulator const &accumulator,
        ElementVector     const &source,
        ElementCompute          alpha_scalar = ElementCompute(1),
        ElementCompute          beta_scalar  = ElementCompute(1)
    ) const {
        
        ElementVector d;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Count; ++i) {
            ElementOutput up_val = params_.up_ptr[i];
            auto gate_i = accumulator[i];
            auto s      = gate_i / (ElementCompute(1) + fast_exp(-gate_i));
            ElementCompute m = up_val * s * params_.alpha * alpha_scalar;
            if (params_.beta != ElementCompute(0)) {
                m += params_.beta * beta_scalar * source[i];
            }
            d[i] = ElementOutput(m);
        }

        return d;
    }
private:
    CUTLASS_DEVICE ElementCompute apply_swish(ElementCompute x) const {
        return x / (ElementCompute(1) + cutlass::fast_exp(-x));
    }
};


template<
  typename ElementOutput,
  int      Count,
  typename ElementAccumulator,
  typename ElementCompute
>
struct SwigluEpilogue : cutlass::epilogue::thread::LinearCombination<
      ElementOutput, Count, ElementAccumulator, ElementCompute> {

    using Base    = cutlass::epilogue::thread::LinearCombination<
                    ElementOutput, Count, ElementAccumulator, ElementCompute
                    >;
    using FragmentAccumulator = typename Base::FragmentAccumulator;
    using FragmentSource      = typename Base::FragmentSource;
    using FragmentOutput      = typename Base::FragmentOutput;

    struct Params : Base::Params {
        ElementOutput const* up_ptr;   // ptr to “up” buffer
        int                   ldm;      // leading dim of up
        int                   batch_stride; // batch stride of up
    };
private:
    Params params_;    
public:
    CUTLASS_HOST_DEVICE
    SwigluEpilogue(Params const &p)
    : Base({p.alpha, p.beta}), params_(p) {}

    CUTLASS_DEVICE
    bool is_source_needed() const { return true; }

    // If you ever split-K, you could implement set_k_partition here.
    CUTLASS_DEVICE
    void set_k_partition(int, int) {}
    using Base::operator();
    // acc   = gate fragment
    // src   = up fragment
    // alpha = your scalar alpha,  beta=unused since is_source_needed==true
    CUTLASS_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const &accum,
        FragmentSource      const &src
    ) const {
      FragmentOutput dst;
      for (int i = 0; i < Count; ++i) {
        ElementCompute g     = accum[i];
        ElementCompute swish = g / (ElementCompute(1) + cutlass::fast_exp(-g));
        dst[i] = ElementOutput(src[i] * swish * params_.alpha);
      }
      return dst;
    }
    CUTLASS_DEVICE
    FragmentOutput
    operator()( FragmentAccumulator const &accum,
                FragmentSource      const &src,
                ElementCompute            alpha,
                ElementCompute            beta   ) const
    {
        FragmentOutput dst;
        for (int i = 0; i < Count; ++i) {
        ElementCompute g     = accum[i];
        ElementCompute swish = g / (ElementCompute(1) + cutlass::fast_exp(-g));
        dst[i] = ElementOutput(src[i] * swish * alpha);
        }
        return dst;
    }

};

template<typename T, typename EpilogueOp>
cutlass::Status run_fused_gemm_batched (
    int M, int N, int K,
    float alpha,
    const T* A, int lda, int batch_stride_A,
    const T* B, int ldb, int batch_stride_B,
    const T* C, int ldc, int batch_stride_C,
    T* out, int ldout, int batch_stride_out,
    float beta,
    int batch_count,
    typename EpilogueOp::Params const &epilogue_params
) {

    using ElementAccumulator = float;
    using Gemm = cutlass::gemm::device::GemmBatched<
        T, cutlass::layout::ColumnMajor,
        T, cutlass::layout::ColumnMajor,
        T, cutlass::layout::ColumnMajor,
        float,                                // accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,
        cutlass::gemm::GemmShape<32,32,32>,
        cutlass::gemm::GemmShape<16,8,8>,
        EpilogueOp
    >;

    typename Gemm::Arguments args(
        { M, N, K },
        { A,      lda }, batch_stride_A,    // input
        { B,  ldb }, batch_stride_B,        // B-block (gate projection)
        { C, ldc }, batch_stride_C,         // C-block *as source* (up projection)
        { out,    ldout }, batch_stride_out,// D-block = output
        epilogue_params,
        batch_count
    );

    size_t workspace_bytes = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);

    Gemm gemm_op;
    gemm_op.initialize(args, workspace.get());
    auto status = gemm_op();
    return status;
}



