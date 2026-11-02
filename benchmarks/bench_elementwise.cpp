#include <benchmark/benchmark.h>

#include "kael/random.h"
#include "naive_add.h"

namespace {

constexpr std::int64_t kSide = 1000;
constexpr std::int64_t kElements = kSide * kSide;

void BM_NaiveAddContiguous(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    const kael::Tensor b = kael::rand({kSide, kSide});
    for (auto _ : state) {
        kael::Tensor out = kael::bench::naive_add(a, b);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.SetItemsProcessed(state.iterations() * kElements);
}
BENCHMARK(BM_NaiveAddContiguous)->Unit(benchmark::kMillisecond);

void BM_NaiveAddBroadcast(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    const kael::Tensor b = kael::rand({kSide});
    for (auto _ : state) {
        kael::Tensor out = kael::bench::naive_add(a, b);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.SetItemsProcessed(state.iterations() * kElements);
}
BENCHMARK(BM_NaiveAddBroadcast)->Unit(benchmark::kMillisecond);

void BM_NaiveAddTransposed(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide}).T();
    const kael::Tensor b = kael::rand({kSide, kSide});
    for (auto _ : state) {
        kael::Tensor out = kael::bench::naive_add(a, b);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.SetItemsProcessed(state.iterations() * kElements);
}
BENCHMARK(BM_NaiveAddTransposed)->Unit(benchmark::kMillisecond);

} // namespace
