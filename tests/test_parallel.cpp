#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kael/autograd/grad_mode.h"
#include "kael/error.h"
#include "kael/parallel.h"

namespace {

class ParallelThreads : public testing::Test {
protected:
    void SetUp() override {
        previous_ = kael::get_num_threads();
        kael::set_num_threads(4);
    }
    void TearDown() override { kael::set_num_threads(previous_); }

private:
    std::int64_t previous_ = 1;
};

} // namespace

TEST_F(ParallelThreads, VisitsEveryIndexOnce) {
    std::vector<std::atomic<int>> visits(10000);
    kael::parallel_for(0, 10000, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            visits[static_cast<std::size_t>(i)].fetch_add(1);
        }
    });
    for (const auto& count : visits) {
        EXPECT_EQ(count.load(), 1);
    }
}

TEST_F(ParallelThreads, HonoursNonZeroBegin) {
    std::atomic<std::int64_t> total{0};
    kael::parallel_for(100, 200, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            total.fetch_add(i);
        }
    });
    EXPECT_EQ(total.load(), (100 + 199) * 100 / 2);
}

TEST_F(ParallelThreads, SmallRangeRunsOnCallingThread) {
    const auto caller = std::this_thread::get_id();
    std::thread::id runner;
    kael::parallel_for(0, 10, 100,
                       [&](std::int64_t, std::int64_t) { runner = std::this_thread::get_id(); });
    EXPECT_EQ(runner, caller);
}

TEST_F(ParallelThreads, RangeIsSplitWhenLargerThanGrain) {
    std::atomic<int> calls{0};
    kael::parallel_for(0, 1000, 10, [&](std::int64_t begin, std::int64_t end) {
        EXPECT_LE(end - begin, 1000);
        calls.fetch_add(1);
    });
    EXPECT_GT(calls.load(), 1);
}

TEST_F(ParallelThreads, RespectsGrain) {
    kael::parallel_for(0, 1000, 300, [&](std::int64_t begin, std::int64_t end) {
        EXPECT_TRUE(end - begin >= 300 || end == 1000) << begin << " " << end;
    });
}

TEST_F(ParallelThreads, EmptyRangeDoesNothing) {
    bool called = false;
    kael::parallel_for(5, 5, 1, [&](std::int64_t, std::int64_t) { called = true; });
    EXPECT_FALSE(called);
}

TEST_F(ParallelThreads, ExceptionReachesCaller) {
    EXPECT_THROW(
        kael::parallel_for(0, 1000, 1,
                           [](std::int64_t, std::int64_t) { throw std::runtime_error("boom"); }),
        std::runtime_error);
}

TEST_F(ParallelThreads, PoolKeepsWorkingAfterAnException) {
    EXPECT_ANY_THROW(kael::parallel_for(
        0, 100, 1, [](std::int64_t, std::int64_t) { throw std::runtime_error("boom"); }));
    std::atomic<std::int64_t> count{0};
    kael::parallel_for(0, 100, 1,
                       [&](std::int64_t begin, std::int64_t end) { count.fetch_add(end - begin); });
    EXPECT_EQ(count.load(), 100);
}

TEST_F(ParallelThreads, GradModeIsCopiedToWorkers) {
    const kael::autograd::NoGradGuard guard;
    std::atomic<int> enabled{0};
    kael::parallel_for(0, 1000, 1, [&](std::int64_t, std::int64_t) {
        if (kael::autograd::GradMode::is_enabled()) {
            enabled.fetch_add(1);
        }
    });
    EXPECT_EQ(enabled.load(), 0);
}

TEST_F(ParallelThreads, NestedCallRunsInline) {
    std::atomic<std::int64_t> count{0};
    kael::parallel_for(0, 16, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            kael::parallel_for(0, 100, 1, [&](std::int64_t inner_begin, std::int64_t inner_end) {
                count.fetch_add(inner_end - inner_begin);
            });
        }
    });
    EXPECT_EQ(count.load(), 1600);
}

TEST_F(ParallelThreads, ConcurrentCallersAllComplete) {
    std::atomic<std::int64_t> count{0};
    std::vector<std::thread> callers;
    for (int caller = 0; caller < 4; ++caller) {
        callers.emplace_back([&] {
            for (int repeat = 0; repeat < 50; ++repeat) {
                kael::parallel_for(0, 1000, 1, [&](std::int64_t begin, std::int64_t end) {
                    count.fetch_add(end - begin);
                });
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
    EXPECT_EQ(count.load(), 4 * 50 * 1000);
}

TEST_F(ParallelThreads, ThreadCountIsReadableInsideBody) {
    std::atomic<std::int64_t> seen{0};
    kael::parallel_for(0, 100, 1,
                       [&](std::int64_t, std::int64_t) { seen.store(kael::get_num_threads()); });
    EXPECT_EQ(seen.load(), 4);
}

TEST_F(ParallelThreads, ChangingThreadCountInsideBodyThrows) {
    EXPECT_THROW(
        kael::parallel_for(0, 100, 1, [](std::int64_t, std::int64_t) { kael::set_num_threads(2); }),
        kael::Error);
}

TEST_F(ParallelThreads, SecondCallerDoesNotWaitForTheActiveJob) {
    std::atomic<std::int64_t> visited{0};
    kael::parallel_for(0, 8, 1, [&](std::int64_t first, std::int64_t) {
        if (first == 0) {
            std::thread other([&] {
                kael::parallel_for(0, 100, 1, [&](std::int64_t from, std::int64_t to) {
                    visited.fetch_add(to - from);
                });
            });
            other.join();
        }
    });
    EXPECT_EQ(visited.load(), 100);
}

TEST(ParallelConfig, ThreadCountIsConfigurable) {
    const std::int64_t previous = kael::get_num_threads();
    kael::set_num_threads(2);
    EXPECT_EQ(kael::get_num_threads(), 2);
    kael::set_num_threads(previous);
    EXPECT_THROW(kael::set_num_threads(0), kael::Error);
}

TEST(ParallelConfig, SingleThreadRunsWholeRangeOnce) {
    const std::int64_t previous = kael::get_num_threads();
    kael::set_num_threads(1);
    int calls = 0;
    kael::parallel_for(0, 1000, 1, [&](std::int64_t begin, std::int64_t end) {
        EXPECT_EQ(end - begin, 1000);
        ++calls;
    });
    EXPECT_EQ(calls, 1);
    kael::set_num_threads(previous);
}
