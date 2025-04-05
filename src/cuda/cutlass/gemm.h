#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/core_io.h>
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
