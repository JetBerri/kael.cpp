#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/random.h"

#include "support/allclose.h"
#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::test::allclose;
using kael::test::values;

namespace {
const float kInf = std::numeric_limits<float>::infinity();
}

TEST(LogSumExp, DoesNotOverflow) {
    auto out = kael::logsumexp(kael::from_vector({1000.0, 1000.0}), 0);
    EXPECT_NEAR(out.item<double>(), 1000.0 + std::log(2.0), 1e-9);
}

TEST(LogSumExp, AllMinusInfinityIsMinusInfinity) {
    EXPECT_EQ(kael::logsumexp(kael::from_vector({-kInf, -kInf}), 0).item<float>(), -kInf);
}

TEST(LogSumExp, KeepdimAndShapes) {
    auto x = kael::zeros({2, 3});
    EXPECT_EQ(kael::logsumexp(x, 1).shape(), (Shape{2}));
    EXPECT_EQ(kael::logsumexp(x, 1, true).shape(), (Shape{2, 1}));
    EXPECT_EQ(kael::logsumexp(kael::zeros({0, 3}), 1).shape(), (Shape{0}));
    EXPECT_EQ(kael::logsumexp(kael::zeros({0}), 0).item<float>(), -kInf);
}

TEST(Softmax, Example) {
    auto out = kael::softmax(kael::from_vector({1.f, 2.f, 3.f}), 0);
    EXPECT_TRUE(allclose(out, kael::from_vector({0.0900306f, 0.244728f, 0.665241f})));
}

TEST(Softmax, RowsSumToOne) {
    kael::manual_seed(0);
    auto rows = kael::sum(kael::softmax(kael::randn({4, 5}), 1), {1});
    EXPECT_TRUE(allclose(rows, kael::ones({4})));
}

TEST(Softmax, LargeValuesDoNotOverflow) {
    auto v = values<float>(kael::softmax(kael::from_vector({1000.f, 1000.f}), 0));
    EXPECT_EQ(v, (std::vector<float>{0.5f, 0.5f}));
}

TEST(Softmax, IntegersBecomeFloat32) {
    auto ints = kael::from_vector(std::vector<std::int64_t>{1, 1});
    EXPECT_EQ(kael::softmax(ints, 0).dtype(), DType::F32);
}

TEST(LogSoftmax, IsExactWhereLogOfSoftmaxIsNot) {
    auto x = kael::from_vector({-1000.f, 0.f});
    EXPECT_NEAR(kael::log_softmax(x, 0).at<float>({0}), -1000.f, 1e-2f);
    EXPECT_EQ(kael::log(kael::softmax(x, 0)).at<float>({0}), -kInf);
}

TEST(LogSoftmax, ExpMatchesSoftmax) {
    kael::manual_seed(1);
    auto x = kael::randn({3, 4});
    EXPECT_TRUE(allclose(kael::exp(kael::log_softmax(x, 1)), kael::softmax(x, 1)));
}

TEST(LogSoftmax, EmptyKeepsShape) {
    EXPECT_EQ(kael::log_softmax(kael::zeros({0, 3}), 1).shape(), (Shape{0, 3}));
    EXPECT_EQ(kael::softmax(kael::zeros({0, 3}), 1).shape(), (Shape{0, 3}));
}

TEST(NnFunctions, InvalidDimThrows) {
    EXPECT_THROW(kael::softmax(kael::zeros({2}), 1), kael::Error);
    EXPECT_THROW(kael::log_softmax(kael::zeros({2}), -2), kael::Error);
    EXPECT_THROW(kael::logsumexp(kael::zeros({2}), 3), kael::Error);
}
