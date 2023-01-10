from dataclasses import dataclass, asdict
import timeit
import sys
import json
from itertools import product

import numpy


@dataclass(frozen=True)
class BenchType:
    double: bool
    n_d: int
    backend: str

    def name(self) -> str:
        double = "Double" if self.double else "Single"
        return f"Benchmark{self.backend.title()}{double}{self.n_d}D"


@dataclass(frozen=True, kw_only=True)
class BenchResult(BenchType):
    log2n: int
    n_slices_per_call: int
    times: numpy.ndarray

    def to_json(self, **kwargs) -> str:
        d = asdict(self)
        d['times'] = list(d['times'])
        return json.dumps(d, **kwargs)


try:
    from ._fft_accel import BenchmarkAccelerateSingle1D, BenchmarkAccelerateDouble1D, BenchmarkAccelerateSingle2D, BenchmarkAccelerateDouble2D
except ImportError:
    ACCEL_BENCHES = {}
else:
    ACCEL_BENCHES = {
        BenchType(False, 1, 'accelerate'): BenchmarkAccelerateSingle1D,
        BenchType(True, 1, 'accelerate'): BenchmarkAccelerateDouble1D,
        BenchType(False, 2, 'accelerate'): BenchmarkAccelerateSingle2D,
        BenchType(True, 2, 'accelerate'): BenchmarkAccelerateDouble2D,
    }



try:
    from ._fft_cuda import BenchmarkCudaSingle1D, BenchmarkCudaDouble1D, BenchmarkCudaSingle2D, BenchmarkCudaDouble2D
except ImportError:
    CUDA_BENCHES = {}
else:
    CUDA_BENCHES = {
        BenchType(False, 1, 'cuda'): BenchmarkCudaSingle1D,
        BenchType(True, 1, 'cuda'): BenchmarkCudaDouble1D,
        BenchType(False, 2, 'cuda'): BenchmarkCudaSingle2D,
        BenchType(True, 2, 'cuda'): BenchmarkCudaDouble2D,
    }


class BenchmarkNumpy():
    def __init__(self, log2n: int, n_slices: int, n_d, dtype):
        self.log2n = log2n
        self.n_slices = n_slices
        self.n_d = n_d

        self.arr = numpy.zeros((2**log2n,) * self.n_d, dtype=dtype)

    def run(self):
        for i in range(self.n_slices):
            self.arr = numpy.fft.fftn(self.arr)
            self.arr = numpy.fft.ifftn(self.arr)

class BenchmarkNumpySingle1D(BenchmarkNumpy):
    def __init__(self, log2n: int, n_slices: int):
        super().__init__(log2n, n_slices, 1, numpy.complex64)

class BenchmarkNumpyDouble1D(BenchmarkNumpy):
    def __init__(self, log2n: int, n_slices: int):
        super().__init__(log2n, n_slices, 1, numpy.complex128)

class BenchmarkNumpySingle2D(BenchmarkNumpy):
    def __init__(self, log2n: int, n_slices: int):
        super().__init__(log2n, n_slices, 2, numpy.complex64)

class BenchmarkNumpyDouble2D(BenchmarkNumpy):
    def __init__(self, log2n: int, n_slices: int):
        super().__init__(log2n, n_slices, 2, numpy.complex128)

NUMPY_BENCHES = {
    BenchType(False, 1, 'numpy'): BenchmarkNumpySingle1D,
    BenchType(True, 1, 'numpy'): BenchmarkNumpyDouble1D,
    BenchType(False, 2, 'numpy'): BenchmarkNumpySingle2D,
    BenchType(True, 2, 'numpy'): BenchmarkNumpyDouble2D,
}

BENCHES = {
    **NUMPY_BENCHES,
    **ACCEL_BENCHES,
    **CUDA_BENCHES,
}

def main(argv=None):
    for (log2n, n_slices) in product([7, 8, 9, 10, 11, 12], [1, 2, 4, 6]):
        for ty, bench in BENCHES.items():
            if ty.backend != 'cuda' and n_slices != 2:
                continue
            print(f"{ty.name()} n={'x'.join([str(1 << log2n)] * ty.n_d)} n_slices={n_slices}", file=sys.stderr, flush=True)
            bench = bench(log2n, n_slices)

            timer = timeit.Timer(lambda: bench.run())
            vec = numpy.array(timer.repeat(repeat=5, number=100 // n_slices))
            total_n = (100 // n_slices) * n_slices

            times_per_slice = vec / total_n
            #print(times_per_slice)

            result = BenchResult(double=ty.double, n_d = ty.n_d, backend=ty.backend, log2n=log2n, n_slices_per_call=n_slices, times=times_per_slice)
            print(result.to_json(), flush=True)


if __name__ == '__main__':
    main()
