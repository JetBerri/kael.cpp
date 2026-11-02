#include <benchmark/benchmark.h>

#include "kael/factory.h"
#include "kael/nn/layernorm.h"
#include "kael/ops.h"
#include "kael/random.h"

namespace {

constexpr std::int64_t kSide = 1000;

template <typename Compute> void run(benchmark::State& state, Compute compute) {
    for (auto _ : state) {
        kael::Tensor out = compute();
        benchmark::DoNotOptimize(out.data_ptr_raw());
    }
}

void BM_CompareAdd(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    const kael::Tensor b = kael::rand({kSide, kSide});
    run(state, [&] { return a + b; });
}
BENCHMARK(BM_CompareAdd)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareAddBroadcast(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    const kael::Tensor b = kael::rand({kSide});
    run(state, [&] { return a + b; });
}
BENCHMARK(BM_CompareAddBroadcast)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareSum(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    run(state, [&] { return kael::sum(a); });
}
BENCHMARK(BM_CompareSum)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareSumFirstDim(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    run(state, [&] { return kael::sum(a, {0}); });
}
BENCHMARK(BM_CompareSumFirstDim)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareSumLastDim(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    run(state, [&] { return kael::sum(a, {1}); });
}
BENCHMARK(BM_CompareSumLastDim)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareSoftmax(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    run(state, [&] { return kael::softmax(a, -1); });
}
BENCHMARK(BM_CompareSoftmax)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareLogSoftmax(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    run(state, [&] { return kael::log_softmax(a, -1); });
}
BENCHMARK(BM_CompareLogSoftmax)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareLayerNorm(benchmark::State& state) {
    const kael::Tensor a = kael::rand({kSide, kSide});
    kael::nn::LayerNorm norm({kSide});
    run(state, [&] { return norm.forward(a); });
}
BENCHMARK(BM_CompareLayerNorm)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareMatmul(benchmark::State& state) {
    const kael::Tensor a = kael::rand({1024, 1024});
    const kael::Tensor b = kael::rand({1024, 1024});
    run(state, [&] { return kael::matmul(a, b); });
}
BENCHMARK(BM_CompareMatmul)->Unit(benchmark::kMicrosecond)->UseRealTime();

} // namespace
