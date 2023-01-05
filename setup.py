import sys
import os.path
from setuptools import setup
from setuptools.config import read_configuration
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
	Pybind11Extension("fft_bench._fft_bench",
		["fft_bench/main.cpp"],
		extra_compile_args=[
			"-std=c++17",
			"-Wall", "-Wextra", #"-Werror",
			"-Wno-error=unused-parameter",
			"-Wno-error=unused-variable",
		],
	),
]

setup(
	ext_modules=ext_modules,
	cmdclass={"build_ext": build_ext},
	zip_safe=False,
)
