#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>
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

template<typename ArchTag_>
struct GemmReluConfig {
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

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float
    >;
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
cutlass::Status run_gemm_relu(
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
            status = gemm_op.initialize(arguments, workspace.get());
            
        }
        return status;
}



