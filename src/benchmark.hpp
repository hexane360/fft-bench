
class Benchmark {
    public:
        size_t log2n;
        size_t nrep;

        Benchmark(size_t log2n, size_t nrep) : log2n(log2n), nrep(nrep) { }
        virtual ~Benchmark() { }
        virtual void run() = 0;
};
