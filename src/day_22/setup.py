import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get absolute path of the 'cuda' directory
cuda_include_path = os.path.abspath("cuda")

# Use dynamically retrieved PyTorch include paths
import torch.utils.cpp_extension
torch_include_paths = torch.utils.cpp_extension.include_paths()

setup(
    name="feed_forward",
    ext_modules=[
        CUDAExtension(
            "feed_forward",
            ["../day_22/feed_forward.cu"],
            include_dirs=[cuda_include_path] + torch_include_paths,
            libraries=["cublas", "cublasLt"],
            extra_compile_args={
                "cxx": [f"-I{cuda_include_path}"] + [f"-I{p}" for p in torch_include_paths],  # Pass to g++
                "nvcc": [f"-I{cuda_include_path}"] + [f"-I{p}" for p in torch_include_paths] + ["-O3"] # Pass to nvcc
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)