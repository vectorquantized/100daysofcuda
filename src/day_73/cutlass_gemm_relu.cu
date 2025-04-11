#include <cuda_runtime.h>
#include "../cuda/cutlass/gemm.h"
#include "../cuda/cutlass/util.h"
#include "../cuda/cuda_utils.h"
#include "../csrc/utils.h"

int main(int argc, char* argv[]) {

    int M = 5120;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0f;
    float beta = 0.0;

    auto init_policy_A = std::make_unique<UniformInitPolicyKernel<cutlass::half_t, cutlass::layout::RowMajor>>(
        2080, -4.0_hf, 4.0_hf, 2);
        
    auto init_policy_B = std::make_unique<UniformInitPolicyKernel<cutlass::half_t, cutlass::layout::RowMajor>>(
        2081, -4.0_hf, 4.0_hf, 2);
        
    auto init_policy_C = std::make_unique<UniformInitPolicyKernel<cutlass::half_t, cutlass::layout::RowMajor>>(
        2082, -4.0_hf, 4.0_hf, 2);

    using AmpereReluConfig = GemmReluConfig<cutlass::arch::Sm80>;
    cutlass::Status status = run_gemm_relu<float, AmpereReluConfig>(
        M, N, K, 
        alpha, beta,
        init_policy_A.get(),
        init_policy_B.get(),
        init_policy_C.get()
    );
    
    CUTLASS_CHECK(status);
    return 0;
}