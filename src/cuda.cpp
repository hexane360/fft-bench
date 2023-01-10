#include "stdio.h"
#include <valarray>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "benchmark.hpp"

namespace py = pybind11;

template <typename T, size_t N = 1>
class BenchmarkCuda : Benchmark {
    public:
        std::valarray<std::complex<T>> arr;
        cufftHandle plan;
        void *devArr;

        BenchmarkCuda(size_t log2n, size_t nrep) : Benchmark(log2n, nrep), arr(0., 1 << (N * log2n)) {
            cudaMalloc(&devArr, sizeof(std::complex<T>) * arr.size());
            if (cudaGetLastError() != cudaSuccess) {
                fprintf(stderr, "CUDA: Failed to allocate on device\n");
            }
            cufftResult_t result = CUFFT_SUCCESS;
            static_assert(N == 1 || N == 2);
            if (N == 1) {
                result = cufftPlan1d(&plan, arr.size(), CUFFT_C2C, 1);
            } else {
                result = cufftPlan2d(&plan, 1 << log2n, 1 << log2n, CUFFT_C2C);
            }
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "CUDA: Failed to create FFT plan\n");
            }
        }
        BenchmarkCuda(BenchmarkCuda&) = delete;
        ~BenchmarkCuda() {
            cufftDestroy(plan);
            cudaFree(devArr);
        }

        void run() {
            cudaMemcpy((void*) devArr, (void*) &arr[0], sizeof(std::complex<T>) * arr.size(), cudaMemcpyHostToDevice);
            for (size_t i = 0; i < nrep; i++) {
                static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value);
                if (std::is_same<T, double>::value) {
                    // double
                    cufftDoubleComplex *ptr = (cufftDoubleComplex*) devArr;
                    cufftExecZ2Z(plan, ptr, ptr, CUFFT_FORWARD);
                    cufftExecZ2Z(plan, ptr, ptr, CUFFT_INVERSE);
                } else if (std::is_same<T, float>::value) {
                    // single
                    cufftComplex *ptr = (cufftComplex*) devArr;
                    cufftExecC2C(plan, ptr, ptr, CUFFT_FORWARD);
                    cufftExecC2C(plan, ptr, ptr, CUFFT_INVERSE);
                }
            }
            cudaMemcpy((void*) &arr[0], (void*) devArr, sizeof(std::complex<T>) * arr.size(), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
};


PYBIND11_MODULE(_fft_cuda, m) {
    m.doc() = "CUDA/cuFFT";

    py::class_<BenchmarkCuda<double, 1>>(m, "BenchmarkCudaDouble1D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkCuda<double, 1>::run);

    py::class_<BenchmarkCuda<double, 2>>(m, "BenchmarkCudaDouble2D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkCuda<double, 2>::run);

    py::class_<BenchmarkCuda<float, 1>>(m, "BenchmarkCudaSingle1D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkCuda<float, 1>::run);

    py::class_<BenchmarkCuda<float, 2>>(m, "BenchmarkCudaSingle2D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkCuda<float, 2>::run);
}
