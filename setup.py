"""
PolarEngine vLLM Plugin -- setup.py

Provides the ``polarengine`` quantization method to vLLM via
the ``vllm.general_plugins`` entry point.
"""

from setuptools import setup, find_packages

setup(
    name="polarengine-vllm",
    version="0.1.0",
    description="PolarEngine quantization plugin for vLLM",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="PolarEngine Team",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "vllm>=0.8.0",
        "triton>=2.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "polarengine = polarengine_vllm:register_polar_quant",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
