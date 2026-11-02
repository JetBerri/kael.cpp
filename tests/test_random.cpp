#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/random.h"

#include "support/values.h"

using kael::DType;
using kael::test::values;

TEST(Random, SameSeedSameValues) {
    kael::manual_seed(123);
    auto a = values<float>(kael::rand({100}));
    auto an = values<double>(kael::randn({50}, DType::F64));
    kael::manual_seed(123);
    auto b = values<float>(kael::rand({100}));
    auto bn = values<double>(kael::randn({50}, DType::F64));
    EXPECT_EQ(a, b);
    EXPECT_EQ(an, bn);
}

TEST(Random, DifferentSeedsDifferentValues) {
    kael::manual_seed(1);
    auto a = values<float>(kael::rand({100}));
    kael::manual_seed(2);
    auto b = values<float>(kael::rand({100}));
    EXPECT_NE(a, b);
}

TEST(Random, Float32AndFloat64SeeTheSameSequence) {
    kael::manual_seed(7);
    auto f = values<float>(kael::randn({20}, DType::F32));
    kael::manual_seed(7);
    auto d = values<double>(kael::randn({20}, DType::F64));
    for (std::size_t i = 0; i < f.size(); ++i) {
        EXPECT_EQ(f[i], static_cast<float>(d[i]));
    }
}

TEST(Random, RandInUnitInterval) {
    kael::manual_seed(0);
    for (DType d : {DType::F32, DType::F64}) {
        auto t = kael::rand({10000}, d);
        kael::dispatch(d, [&]<typename T>() {
            for (T v : values<T>(t)) {
                ASSERT_GE(v, T{0});
                ASSERT_LT(v, T{1});
            }
        });
    }
}

TEST(Random, RandnMeanAndStd) {
    kael::manual_seed(0);
    auto v = values<double>(kael::randn({100000}, DType::F64));
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    const double mean = sum / static_cast<double>(v.size());
    double sq = 0.0;
    for (double x : v) {
        sq += (x - mean) * (x - mean);
    }
    const double stddev = std::sqrt(sq / static_cast<double>(v.size()));
    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(stddev, 1.0, 0.02);
}

TEST(Random, RandintRespectsBounds) {
    kael::manual_seed(0);
    auto t = kael::randint(-3, 4, {5000});
    EXPECT_EQ(t.dtype(), DType::I64);
    auto v = values<std::int64_t>(t);
    EXPECT_EQ(*std::min_element(v.begin(), v.end()), -3);
    EXPECT_EQ(*std::max_element(v.begin(), v.end()), 3);
    EXPECT_THROW(kael::randint(2, 2, {1}), kael::Error);
}

TEST(Random, RandpermIsAPermutation) {
    kael::manual_seed(0);
    auto v = values<std::int64_t>(kael::randperm(50));
    std::sort(v.begin(), v.end());
    for (std::int64_t i = 0; i < 50; ++i) {
        EXPECT_EQ(v[static_cast<std::size_t>(i)], i);
    }
}

TEST(Random, NonFloatingDtypesThrow) {
    EXPECT_THROW(kael::rand({2}, DType::I64), kael::Error);
    EXPECT_THROW(kael::rand({2}, DType::Bool), kael::Error);
    EXPECT_THROW(kael::randn({2}, DType::I64), kael::Error);
}

TEST(Random, BernoulliRespectsProbabilityExtremes) {
    EXPECT_EQ(kael::sum(kael::bernoulli({100}, 0.0)).item(), 0.0);
    EXPECT_EQ(kael::sum(kael::bernoulli({100}, 1.0)).item(), 100.0);
}

TEST(Random, BernoulliMatchesProbability) {
    kael::manual_seed(31);
    EXPECT_NEAR(kael::mean(kael::bernoulli({100000}, 0.3)).item(), 0.3, 0.01);
}

TEST(Random, BernoulliRejectsInvalidProbability) {
    EXPECT_THROW((void)kael::bernoulli({1}, 1.5), kael::Error);
    EXPECT_THROW((void)kael::bernoulli({1}, -0.5), kael::Error);
}

TEST(RandomThreads, ConcurrentSamplingDoesNotRace) {
    const kael::Tensor weights = kael::ones({8});
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&] {
            for (int round = 0; round < 50; ++round) {
                (void)kael::rand({16});
                (void)kael::randn({16});
                (void)kael::randint(0, 10, {16});
                (void)kael::randperm(16);
                (void)kael::multinomial(weights, 3);
                kael::manual_seed(static_cast<std::uint64_t>(round));
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    SUCCEED();
}
