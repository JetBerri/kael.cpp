#include <benchmark/benchmark.h>

#include "kael/factory.h"
#include "kael/nn/layernorm.h"
#include "kael/nn/linear.h"
#include "kael/nn/loss.h"
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

template <typename Loss> void cross_entropy_step(benchmark::State& state, Loss loss) {
    const kael::Tensor values = kael::randn({state.range(0), state.range(1)});
    const kael::Tensor targets = kael::randint(0, state.range(1), {state.range(0)});
    for (auto _ : state) {
        kael::Tensor logits = values.detach().requires_grad_();
        loss(logits, targets).backward();
        benchmark::DoNotOptimize(logits.grad()->data_ptr_raw());
    }
}

void BM_CrossEntropyFused(benchmark::State& state) {
    cross_entropy_step(state, [](const kael::Tensor& logits, const kael::Tensor& targets) {
        return kael::nn::cross_entropy(logits, targets);
    });
}
BENCHMARK(BM_CrossEntropyFused)
    ->Args({64, 10})
    ->Args({4096, 65})
    ->Args({1024, 50257})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

void BM_CrossEntropyComposed(benchmark::State& state) {
    cross_entropy_step(state, [](const kael::Tensor& logits, const kael::Tensor& targets) {
        return kael::nn::nll_loss(kael::log_softmax(logits, 1), targets);
    });
}
BENCHMARK(BM_CrossEntropyComposed)
    ->Args({64, 10})
    ->Args({4096, 65})
    ->Args({1024, 50257})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

void linear_step(benchmark::State& state, bool bias) {
    kael::nn::Linear layer(784, 128, bias);
    const kael::Tensor input = kael::randn({64, 784});
    for (auto _ : state) {
        layer.zero_grad();
        kael::sum(layer.forward(input)).backward();
    }
}

void BM_LinearWithBias(benchmark::State& state) {
    linear_step(state, true);
}
BENCHMARK(BM_LinearWithBias)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_LinearWithoutBias(benchmark::State& state) {
    linear_step(state, false);
}
BENCHMARK(BM_LinearWithoutBias)->Unit(benchmark::kMicrosecond)->UseRealTime();

void BM_CompareMatmul(benchmark::State& state) {
    const kael::Tensor a = kael::rand({1024, 1024});
    const kael::Tensor b = kael::rand({1024, 1024});
    run(state, [&] { return kael::matmul(a, b); });
}
BENCHMARK(BM_CompareMatmul)->Unit(benchmark::kMicrosecond)->UseRealTime();

} // namespace
