#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "kael/iterator.h"
#include "kael/kael.h"

using kael::Tensor;

namespace {

constexpr std::int64_t kLargeCount = std::int64_t{1} << 21;

class ThreadCount {
public:
    explicit ThreadCount(std::int64_t count) : previous_(kael::get_num_threads()) {
        kael::set_num_threads(count);
    }
    ThreadCount(const ThreadCount&) = delete;
    ThreadCount& operator=(const ThreadCount&) = delete;
    ~ThreadCount() { kael::set_num_threads(previous_); }

private:
    std::int64_t previous_;
};

template <typename F> void expect_same_for_any_thread_count(F compute) {
    const Tensor reference = [&] {
        const ThreadCount threads(1);
        return compute();
    }();
    for (const std::int64_t count : {2, 3, 8}) {
        const ThreadCount threads(count);
        EXPECT_TRUE(kael::equal(compute(), reference)) << count << " threads";
    }
}

} // namespace

TEST(ParallelIterator, LargeContiguousRangeIsSplit) {
    const ThreadCount threads(4);
    const Tensor input = kael::zeros({kLargeCount});
    kael::TensorIterator iterator;
    iterator.allocate_output(kael::DType::F32).add_input(input).build();
    std::atomic<std::int64_t> calls{0};
    std::atomic<std::int64_t> visited{0};
    iterator.run([&](std::byte**, const std::int64_t*, std::int64_t count) {
        ++calls;
        visited += count;
    });
    EXPECT_GT(calls.load(), 1);
    EXPECT_EQ(visited.load(), kLargeCount);
}

TEST(ParallelIterator, SmallRangeStaysInOneCall) {
    const ThreadCount threads(4);
    const Tensor input = kael::zeros({1000});
    kael::TensorIterator iterator;
    iterator.allocate_output(kael::DType::F32).add_input(input).build();
    std::int64_t calls = 0;
    iterator.run([&](std::byte**, const std::int64_t*, std::int64_t) { ++calls; });
    EXPECT_EQ(calls, 1);
}

TEST(ParallelIterator, EveryLayoutVisitsEveryElementInItsPlace) {
    const ThreadCount threads(4);
    const std::int64_t rows = 300;
    const std::int64_t columns = 400;
    const std::int64_t depth = 7;
    Tensor base = kael::arange(0, rows * columns * depth).to(kael::DType::F64);
    base = base.reshape({rows, columns, depth});

    const std::vector<Tensor> layouts = {
        base,
        base.permute({2, 0, 1}),
        base.transpose(0, 1),
        base.slice(1, 1, columns, 2),
    };
    for (const Tensor& layout : layouts) {
        const Tensor doubled = layout * 2;
        for (const std::int64_t i : {std::int64_t{0}, layout.size(0) / 2, layout.size(0) - 1}) {
            for (const std::int64_t j : {std::int64_t{0}, layout.size(1) - 1}) {
                const std::int64_t k = layout.size(2) - 1;
                EXPECT_EQ(doubled.at<double>({i, j, k}), layout.at<double>({i, j, k}) * 2);
            }
        }
    }
}

TEST(ParallelIterator, ResultsDoNotDependOnThreadCount) {
    kael::manual_seed(11);
    const Tensor a = kael::randn({1024, 1024});
    const Tensor b = kael::randn({1024});
    expect_same_for_any_thread_count([&] { return a + b; });
    expect_same_for_any_thread_count([&] { return kael::exp(a.T()); });
    expect_same_for_any_thread_count([&] { return a.T().contiguous(); });
}

TEST(ParallelIterator, IntegerDivisionByZeroStillThrows) {
    const ThreadCount threads(4);
    const Tensor a = kael::ones({kLargeCount}, kael::DType::I64);
    const Tensor b = kael::zeros({kLargeCount}, kael::DType::I64);
    EXPECT_THROW(kael::floor_divide(a, b), kael::Error);
}

TEST(ParallelReduce, ResultsDoNotDependOnThreadCount) {
    kael::manual_seed(12);
    const Tensor x = kael::randn({1024, 1024}, kael::DType::F64);
    expect_same_for_any_thread_count([&] { return kael::sum(x); });
    expect_same_for_any_thread_count([&] { return kael::mean(x); });
    expect_same_for_any_thread_count([&] { return kael::var(x); });
    expect_same_for_any_thread_count([&] { return kael::norm(x); });
    expect_same_for_any_thread_count([&] { return kael::prod(x); });
    expect_same_for_any_thread_count([&] { return kael::sum(x, {1}); });
    expect_same_for_any_thread_count([&] { return kael::max(x, {0}); });
    expect_same_for_any_thread_count([&] { return kael::argmax(x, 1); });
}

TEST(ParallelReduce, ManyOutputsMatchTheirSliceByHand) {
    const ThreadCount threads(4);
    kael::manual_seed(13);
    const Tensor x = kael::randn({2000, 300}, kael::DType::F64);
    const Tensor totals = kael::sum(x, {1});
    for (const std::int64_t row : {std::int64_t{0}, std::int64_t{999}, std::int64_t{1999}}) {
        double expected = 0.0;
        for (std::int64_t column = 0; column < 300; ++column) {
            expected += x.at<double>({row, column});
        }
        EXPECT_NEAR(totals.at<double>({row}), expected, 1e-11);
    }
}

TEST(ParallelReduce, LongSumIsMoreAccurateThanRunningTotal) {
    const std::int64_t count = std::int64_t{1} << 24;
    const Tensor x = kael::full({count}, 0.1, kael::DType::F64);

    double running = 0.0;
    const double* values = x.data_ptr<double>();
    for (std::int64_t i = 0; i < count; ++i) {
        running += values[i];
    }
    const double exact = 0.1 * static_cast<double>(count);
    EXPECT_LT(std::abs(kael::sum(x).item<double>() - exact), std::abs(running - exact));
}

TEST(ParallelReduce, EmptyReductionsKeepTheirMeaning) {
    const Tensor empty = kael::zeros({0, 4});
    EXPECT_EQ(kael::sum(empty).item<float>(), 0.0f);
    EXPECT_EQ(kael::prod(empty).item<float>(), 1.0f);
    EXPECT_TRUE(std::isnan(kael::mean(empty).item<float>()));
    EXPECT_EQ(kael::sum(empty, {0}).numel(), 4);
}
