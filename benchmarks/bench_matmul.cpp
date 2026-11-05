#include <cstdint>

#include <benchmark/benchmark.h>

#include "kael/factory.h"
#include "kael/matmul_versions.h"
#include "kael/ops.h"
#include "kael/parallel.h"
#include "kael/random.h"

namespace {

using kael::matmul_versions::Sizes;

template <typename T> using Version = void (*)(const T*, const T*, T*, const Sizes&);

void report_flops(benchmark::State& state, std::int64_t n) {
    state.counters["flops"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 *
                                                     static_cast<double>(n * n * n),
                                                 benchmark::Counter::kIsRate);
}

void BM_Matmul(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const kael::Tensor a = kael::rand({n, n});
    const kael::Tensor b = kael::rand({n, n});
    for (auto _ : state) {
        kael::Tensor out = kael::matmul(a, b);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    report_flops(state, n);
}
BENCHMARK(BM_Matmul)->Arg(128)->Arg(256)->Arg(512)->Unit(benchmark::kMillisecond)->UseRealTime();

template <Version<float> Multiply> void BM_MatmulVersion(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const kael::Tensor a = kael::rand({n, n});
    const kael::Tensor b = kael::rand({n, n});
    kael::Tensor out = kael::zeros({n, n});
    for (auto _ : state) {
        Multiply(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), {n, n, n});
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    report_flops(state, n);
}

void version_sizes(benchmark::internal::Benchmark* bench) {
    bench->Arg(256)
        ->Arg(512)
        ->Arg(1024)
        ->Arg(2048)
        ->Unit(benchmark::kMillisecond)
        ->MinTime(0.1)
        ->UseRealTime();
}

using namespace kael::matmul_versions;
BENCHMARK(BM_MatmulVersion<v1_ijk<float>>)->Apply(version_sizes);
BENCHMARK(BM_MatmulVersion<v2_ikj<float>>)->Apply(version_sizes);
BENCHMARK(BM_MatmulVersion<v3_tiled<float>>)->Apply(version_sizes);
BENCHMARK(BM_MatmulVersion<v4_packed<float>>)->Apply(version_sizes);
BENCHMARK(BM_MatmulVersion<v5_simd<float>>)->Apply(version_sizes);
BENCHMARK(BM_MatmulVersion<v6_threaded<float>>)->Apply(version_sizes);

#if defined(KAEL_BENCH_BLAS)
BENCHMARK(BM_MatmulVersion<blas<float>>)->Apply(version_sizes);
#endif

void BM_MatmulThreads(benchmark::State& state) {
    kael::set_num_threads(state.range(1));
    const std::int64_t n = state.range(0);
    const kael::Tensor a = kael::rand({n, n});
    const kael::Tensor b = kael::rand({n, n});
    kael::Tensor out = kael::zeros({n, n});
    for (auto _ : state) {
        v6_threaded<float>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                           {n, n, n});
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    report_flops(state, n);
}
BENCHMARK(BM_MatmulThreads)
    ->ArgsProduct({{1024}, {1, 2, 4, 5, 8, 15}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace
