#include <benchmark/benchmark.h>

#include "kael/ops.h"
#include "kael/random.h"

namespace {

void BM_Matmul(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const kael::Tensor a = kael::rand({n, n});
    const kael::Tensor b = kael::rand({n, n});
    for (auto _ : state) {
        kael::Tensor out = kael::matmul(a, b);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2.0 * static_cast<double>(n * n * n),
        benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Matmul)->Arg(128)->Arg(256)->Arg(512)->Unit(benchmark::kMillisecond);

} // namespace
