cmake_minimum_required(VERSION 3.18...3.24)

# get version from python
set(VERSION_REGEX "__version__ = \"(.+)\"")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/fft_bench/__init__.py"
     VERSION REGEX ${VERSION_REGEX})
string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION "${VERSION}")

project(fft_bench VERSION "${VERSION}" LANGUAGES CXX)

# default PY_BUILD_CMAKE_*_NAME (if cmake is called manually)
if (NOT DEFINED PY_BUILD_CMAKE_MODULE_NAME)
	set(PY_BUILD_CMAKE_MODULE_NAME "fft_bench")
endif()

if (NOT DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
	set(PY_BUILD_CMAKE_PACKAGE_NAME "fft_bench")
endif()

# default build type
if (CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Generated for configurations: ${CMAKE_CONFIGURAITON_TYPES}")
elseif (NOT CMAKE_BUILD_TYPE)
	message(STATUS "Defaulting to build type 'RelWithDebInfo'")
	set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "CMake build type (Debug, Release, RelWithDebInfo, MinSizeRel)." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
	message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

include(cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()

include(CheckLanguage)
check_language(CUDA)

set(targets "")

if(APPLE)
	message(STATUS "Building for Apple Accelerate.")
	pybind11_add_module(_fft_accel MODULE "src/accel.cpp" "src/benchmark.hpp")
	list(APPEND targets _fft_accel)
elseif(CMAKE_CUDA_COMPILER)
	message(STATUS "Building for CUDA.")
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)

	pybind11_add_module(_fft_cuda MODULE "src/cuda.cpp" "src/benchmark.hpp")
	list(APPEND targets _fft_cuda)

	target_link_libraries(_fft_cuda PUBLIC CUDA::cudart CUDA::cufft)
else()
	message(STATUS "CUDA not found. Not building.")
endif()

foreach(target ${targets})
	target_compile_definitions(${target} PRIVATE "VERSION=${VERSION}")
endforeach()

install(TARGETS ${targets}
	EXCLUDE_FROM_ALL
	COMPONENT python_modules
	DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME})
