import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension

# Get absolute paths
this_dir = os.path.dirname(os.path.abspath(__file__))
cuda_include_path = os.path.join(this_dir, "cuda")
cutlass_path = os.environ.get("CUTLASS_PATH", None)
assert cutlass_path is not None
cutlass_include_paths = [f"{cutlass_path}/include", f"{cutlass_path}/tools/util/include"]

# Get PyTorch include paths dynamically
torch_include_paths = torch.utils.cpp_extension.include_paths()

setup(
    name="cblas",
    ext_modules=[
        CUDAExtension(
            "cblas",
            sources=[
                "bindings.cpp",             # Python binding file
                "naive_transpose_cutlass.cu",       # Main implementation
            ],
            include_dirs=[cuda_include_path] + torch_include_paths + cutlass_include_paths,
            libraries=["cublas", "cublasLt"],  # Ensure cuBLAS is linked
            extra_compile_args={
                "cxx": ["-O3"],  # Optimize C++ compilation
                "nvcc": ["-O3"]  # Optimize NVCC compilation
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)