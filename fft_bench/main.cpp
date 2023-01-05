#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Accelerate/Accelerate.h>

//#include <complex>
#include <valarray>
#include <type_traits>
#include <tuple>
#include <stdio.h>

namespace py = pybind11;

class Benchmark {
    public:
        size_t log2n;
        size_t nrep;

        Benchmark(size_t log2n, size_t nrep) : log2n(log2n), nrep(nrep) { }
        virtual ~Benchmark() { }
        virtual void run() = 0;
};

class BenchmarkAccelerateDouble1D : Benchmark {
    public:
        std::valarray<double> real;
        std::valarray<double> imag;
        DSPDoubleSplitComplex ptrs;
        FFTSetupD setup;

        BenchmarkAccelerateDouble1D(size_t log2n, size_t nrep) : Benchmark(log2n, nrep), real(0., 1 << log2n), imag(0., 1 << log2n) {
            setup = vDSP_create_fftsetupD(log2n, 2);
            ptrs = { &real[0], &imag[0] };
        }

        BenchmarkAccelerateDouble1D(BenchmarkAccelerateDouble1D&) = delete;
        ~BenchmarkAccelerateDouble1D() {
            vDSP_destroy_fftsetupD(setup);
        }

        void run() {
            for (size_t i = 0; i < nrep; i++) {
                //printf("iteration %zu\n", i);
                vDSP_fft_zipD(setup, &ptrs, 1, log2n, FFT_FORWARD);
                vDSP_fft_zipD(setup, &ptrs, 1, log2n, FFT_INVERSE);
            }
        }
};


class BenchmarkAccelerateDouble2D : Benchmark {
    public:
        std::valarray<double> real;
        std::valarray<double> imag;
        DSPDoubleSplitComplex ptrs;
        FFTSetupD setup;

        BenchmarkAccelerateDouble2D(size_t log2n, size_t nrep) : Benchmark(log2n, nrep), real(0., 1 << (2*log2n)), imag(0., 1 << (2*log2n)) {
            setup = vDSP_create_fftsetupD(log2n, 2);
            ptrs = { &real[0], &imag[0] };
        }

        BenchmarkAccelerateDouble2D(BenchmarkAccelerateDouble2D&) = delete;
        ~BenchmarkAccelerateDouble2D() {
            vDSP_destroy_fftsetupD(setup);
        }

        void run() {
            for (size_t i = 0; i < nrep; i++) {
                //printf("iteration %zu\n", i);
                vDSP_fft2d_zipD(setup, &ptrs, 1, 0, log2n, log2n, FFT_FORWARD);
                vDSP_fft2d_zipD(setup, &ptrs, 1, 0, log2n, log2n, FFT_INVERSE);
            }
        }
};


class BenchmarkAccelerateSingle1D : Benchmark {
    public:
        std::valarray<float> real;
        std::valarray<float> imag;
        DSPSplitComplex ptrs;
        FFTSetup setup;

        BenchmarkAccelerateSingle1D(size_t log2n, size_t nrep) : Benchmark(log2n, nrep), real(0., 1 << log2n), imag(0., 1 << log2n) {
            setup = vDSP_create_fftsetup(log2n, 2);
            ptrs = { &real[0], &imag[0] };
        }

        BenchmarkAccelerateSingle1D(BenchmarkAccelerateSingle1D&) = delete;
        ~BenchmarkAccelerateSingle1D() {
            vDSP_destroy_fftsetup(setup);
        }

        void run() {
            for (size_t i = 0; i < nrep; i++) {
                //printf("iteration %zu\n", i);
                vDSP_fft_zip(setup, &ptrs, 1, log2n, FFT_FORWARD);
                vDSP_fft_zip(setup, &ptrs, 1, log2n, FFT_INVERSE);
            }
        }
};


class BenchmarkAccelerateSingle2D : Benchmark {
    public:
        std::valarray<float> real;
        std::valarray<float> imag;
        DSPSplitComplex ptrs;
        FFTSetup setup;

        BenchmarkAccelerateSingle2D(size_t log2n, size_t nrep) : Benchmark(log2n, nrep), real(0., 1 << (2*log2n)), imag(0., 1 << (2*log2n)) {
            setup = vDSP_create_fftsetup(log2n, 2);
            ptrs = { &real[0], &imag[0] };
            //printf("ctor\n");
        }

        BenchmarkAccelerateSingle2D(BenchmarkAccelerateSingle2D&) = delete;
        ~BenchmarkAccelerateSingle2D() {
            vDSP_destroy_fftsetup(setup);
            //printf("dtor\n");
        }

        void run() {
            for (size_t i = 0; i < nrep; i++) {
                //printf("iteration %zu\n", i);
                vDSP_fft2d_zip(setup, &ptrs, 1, 0, log2n, log2n, FFT_FORWARD);
                vDSP_fft2d_zip(setup, &ptrs, 1, 0, log2n, log2n, FFT_INVERSE);
            }
        }
};


PYBIND11_MODULE(_fft_bench, m) {
    m.doc() = "Benchmarking utilities";

    //m.def("fft1d", &fft1d, "FFT");

    //py::class_<Benchmark> b(m, "Benchmark");

    py::class_<BenchmarkAccelerateDouble1D>(m, "BenchmarkAccelerateDouble1D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkAccelerateDouble1D::run);

    py::class_<BenchmarkAccelerateDouble2D>(m, "BenchmarkAccelerateDouble2D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkAccelerateDouble2D::run);

    py::class_<BenchmarkAccelerateSingle1D>(m, "BenchmarkAccelerateSingle1D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkAccelerateSingle1D::run);

    py::class_<BenchmarkAccelerateSingle2D>(m, "BenchmarkAccelerateSingle2D")
        .def(py::init<size_t, size_t>())
        .def("run", &BenchmarkAccelerateSingle2D::run);
}