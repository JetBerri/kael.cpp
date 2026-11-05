#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "kael/kael.h"

namespace {

using kael::DType;
using kael::Tensor;

constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();

Tensor longs(std::vector<std::int64_t> values) {
    return kael::from_vector(values);
}

std::vector<std::int64_t> values_of(const Tensor& t) {
    std::vector<std::int64_t> out;
    const Tensor flat = t.contiguous();
    for (std::int64_t i = 0; i < flat.numel(); ++i) {
        out.push_back(flat.data_ptr<std::int64_t>()[i]);
    }
    return out;
}

TEST(IntegerOverflow, AdditionWrapsAround) {
    EXPECT_EQ(values_of(longs({kMax}) + longs({1})), std::vector<std::int64_t>{kMin});
}

TEST(IntegerOverflow, SubtractionWrapsAround) {
    EXPECT_EQ(values_of(longs({kMin}) - longs({1})), std::vector<std::int64_t>{kMax});
}

TEST(IntegerOverflow, MultiplicationWrapsAround) {
    EXPECT_EQ(values_of(longs({kMax}) * longs({2})), std::vector<std::int64_t>{-2});
}

TEST(IntegerOverflow, NegatingTheMinimumWrapsToItself) {
    EXPECT_EQ(values_of(kael::neg(longs({kMin}))), std::vector<std::int64_t>{kMin});
    EXPECT_EQ(values_of(kael::abs(longs({kMin}))), std::vector<std::int64_t>{kMin});
}

TEST(IntegerOverflow, SumAndProductWrapAround) {
    EXPECT_EQ(kael::sum(longs({kMax, 1})).item<std::int64_t>(), kMin);
    EXPECT_EQ(kael::prod(longs({kMax, 2})).item<std::int64_t>(), -2);
}

TEST(IntegerOverflow, MatmulWrapsAround) {
    const Tensor a = kael::from_vector(std::vector<std::int64_t>{kMax, 1}, {1, 2});
    const Tensor b = kael::from_vector(std::vector<std::int64_t>{1, 1}, {2, 1});
    EXPECT_EQ(kael::matmul(a, b).item<std::int64_t>(), kMin);
}

TEST(IntegerOverflow, CumsumWrapsAround) {
    EXPECT_EQ(values_of(kael::cumsum(longs({kMax, 1}), 0)),
              (std::vector<std::int64_t>{kMax, kMin}));
}

TEST(IntegerOverflow, ScatterAddWrapsAround) {
    Tensor target = longs({kMax});
    target.scatter_add_(0, longs({0}), longs({1}));
    EXPECT_EQ(values_of(target), std::vector<std::int64_t>{kMin});
}

} // namespace
