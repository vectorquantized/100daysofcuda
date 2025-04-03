#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

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