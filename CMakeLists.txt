cmake_minimum_required(VERSION 3.18...3.24)

# get version from python
set(VERSION_REGEX "__version__ = \"(.+)\"")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/fft_bench/__init__.py"
     VERSION REGEX ${VERSION_REGEX})
string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION "${VERSION}")

project(fft-bench VERSION ${VERSION} LANGUAGES CXX)

include(cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()

include(CheckLanguage)
check_language(CUDA)

set(targets "")

if(APPLE)
	message(STATUS "Building for Apple Accelerate.")
	pybind11_add_module(_fft_accel MODULE "src/accel.cpp" "src/benchmark.hpp")
	list(APPEND targets _fft_accel)
elseif(CMAKE_CUDA_COMPILER_ID)
	message(STATUS "Building for CUDA.")
	enable_language(CUDA)

	pybind11_add_module(_fft_cuda MODULE "src/cuda.cpp" "src/cuda.cu" "src/benchmark.hpp")
	list(APPEND targets _fft_cuda)
else()
	message(STATUS "CUDA not found. Not building.")
endif()

foreach(target ${targets})
	target_compile_definitions(${target} PRIVATE VERSION="${VERSION}")
endforeach()

install(TARGETS ${targets}
	EXCLUDE_FROM_ALL
	COMPONENT python_modules
	DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
