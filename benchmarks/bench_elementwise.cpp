#include <benchmark/benchmark.h>

#include "kael/kernels.h"
#include "kael/random.h"

#include "naive_add.h"

namespace {

constexpr std::int64_t kSide = 1000;
constexpr std::int64_t kElements = kSide * kSide;
constexpr auto kAdd = [](auto x, auto y) { return x + y; };

void run_kernel_add(benchmark::State& state, const kael::Tensor& a, const kael::Tensor& b) {
    for (auto _ : state) {
        kael::Tensor out = kael::binary_kernel(a, b, kAdd);
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.SetItemsProcessed(state.iterations() * kElements);
}

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

void BM_KernelAddContiguous(benchmark::State& state) {
    run_kernel_add(state, kael::rand({kSide, kSide}), kael::rand({kSide, kSide}));
}
BENCHMARK(BM_KernelAddContiguous)->Unit(benchmark::kMillisecond);

void BM_KernelAddBroadcast(benchmark::State& state) {
    run_kernel_add(state, kael::rand({kSide, kSide}), kael::rand({kSide}));
}
BENCHMARK(BM_KernelAddBroadcast)->Unit(benchmark::kMillisecond);

void BM_KernelAddTransposed(benchmark::State& state) {
    run_kernel_add(state, kael::rand({kSide, kSide}).T(), kael::rand({kSide, kSide}));
}
BENCHMARK(BM_KernelAddTransposed)->Unit(benchmark::kMillisecond);

void BM_ContiguousFromTransposed(benchmark::State& state) {
    const kael::Tensor transposed = kael::rand({kSide, kSide}).T();
    for (auto _ : state) {
        kael::Tensor out = transposed.contiguous();
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
    state.SetItemsProcessed(state.iterations() * kElements);
}
BENCHMARK(BM_ContiguousFromTransposed)->Unit(benchmark::kMillisecond);

} // namespace
