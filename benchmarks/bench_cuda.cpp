#include <cstdint>

#include <benchmark/benchmark.h>

#include "kael/device.h"
#include "kael/ops.h"
#include "kael/random.h"

namespace {

template <typename Compute> void run_on_device(benchmark::State& state, Compute compute) {
    if (!kael::cuda_is_available()) {
        state.SkipWithError("no CUDA device");
        return;
    }
    for (auto _ : state) {
        kael::Tensor out = compute();
        kael::cuda_synchronize();
        benchmark::DoNotOptimize(out.device_data());
    }
}

void BM_DeviceAdd(benchmark::State& state) {
    const kael::Tensor a = kael::rand({state.range(0)}).cuda();
    const kael::Tensor b = kael::rand({state.range(0)}).cuda();
    run_on_device(state, [&] { return a + b; });
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_DeviceAdd)->Arg(1 << 20)->Arg(1 << 24)->UseRealTime();

void BM_DeviceSum(benchmark::State& state) {
    const kael::Tensor a = kael::rand({state.range(0)}).cuda();
    run_on_device(state, [&] { return kael::sum(a); });
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_DeviceSum)->Arg(1 << 20)->Arg(1 << 24)->UseRealTime();

void BM_DeviceMatmul(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const kael::Tensor a = kael::rand({n, n}).cuda();
    const kael::Tensor b = kael::rand({n, n}).cuda();
    run_on_device(state, [&] { return kael::matmul(a, b); });
    state.counters["flops"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 *
                                                     static_cast<double>(n * n * n),
                                                 benchmark::Counter::kIsRate);
}
BENCHMARK(BM_DeviceMatmul)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace
