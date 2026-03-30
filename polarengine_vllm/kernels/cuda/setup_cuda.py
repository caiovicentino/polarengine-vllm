"""Build script for PolarEngine CUDA kernels.

Build (from project root):
    python polarengine_vllm/kernels/cuda/setup_cuda.py install

Or editable (for development):
    pip install -e . -v          # uses the top-level setup.py / pyproject.toml

The extension compiles two CUDA source files into a single module called
``polarengine_cuda`` that exposes:
    - polarengine_cuda.polar_gemv(codes, x, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    - polarengine_cuda.polar_gemv_packed(packed_codes, x, norms, ct_scaled, out_f, in_f_half, n_blocks, block_size)
    - polarengine_cuda.fwht(data, block_size)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="polarengine_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="polarengine_cuda",
            sources=[
                "polarengine_vllm/kernels/cuda/polar_gemv_cuda.cu",
                "polarengine_vllm/kernels/cuda/fwht_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # Target Volta+ (sm_70) through Blackwell (sm_100)
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
