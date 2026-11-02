#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "kael/autograd/gradcheck.h"
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

TEST(LogSoftmax, InfinityGivesNanLikePyTorch) {
    auto v = values<float>(kael::log_softmax(kael::from_vector({kInf, 1.f}), 0));
    EXPECT_TRUE(std::isnan(v[0]));
    EXPECT_TRUE(std::isnan(v[1]));
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

class FusedKernels : public testing::TestWithParam<kael::DType> {};

TEST_P(FusedKernels, LastDimensionMatchesTheComposedPath) {
    kael::manual_seed(21);
    const kael::Tensor x = kael::randn({40, 33}, GetParam());
    const kael::Tensor x_first = x.T().contiguous();
    const kael::test::AllcloseOptions rounding_only{
        .tolerance = GetParam() == kael::DType::F32 ? kael::test::Tolerance{1e-6, 1e-6}
                                                    : kael::test::Tolerance{1e-14, 1e-14}};
    EXPECT_TRUE(
        kael::test::allclose(kael::softmax(x, -1), kael::softmax(x_first, 0).T(), rounding_only));
    EXPECT_TRUE(kael::test::allclose(kael::log_softmax(x, 1), kael::log_softmax(x_first, 0).T(),
                                     rounding_only));
}

TEST_P(FusedKernels, HandlesNonFiniteValuesLikeTheComposedPath) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const kael::Tensor x =
        kael::from_vector<double>({1.0, inf, 2.0, -inf, 1.0, 2.0, nan, 1.0, 2.0}, {3, 3})
            .to(GetParam());
    const kael::Tensor x_first = x.T().contiguous();
    const kael::test::AllcloseOptions options{.equal_nan = true};
    EXPECT_TRUE(kael::test::allclose(kael::softmax(x, -1), kael::softmax(x_first, 0).T(), options));
    EXPECT_TRUE(
        kael::test::allclose(kael::log_softmax(x, -1), kael::log_softmax(x_first, 0).T(), options));
}

TEST_P(FusedKernels, StandardizeMatchesItsDefinition) {
    kael::manual_seed(22);
    const kael::Tensor x = kael::randn({4, 3, 5}, GetParam());
    const kael::Tensor expected =
        (x - kael::mean(x, {-2, -1}, true)) / kael::sqrt(kael::var(x, {-2, -1}, true, 0) + 1e-5);
    EXPECT_TRUE(kael::test::allclose(kael::standardize(x, 2, 1e-5), expected,
                                     {.tolerance = kael::test::Tolerance{1e-4, 1e-4}}));
}

INSTANTIATE_TEST_SUITE_P(Floating, FusedKernels,
                         testing::Values(kael::DType::F32, kael::DType::F64));

TEST(FusedKernels, StridedInputAndWideRowsWork) {
    kael::manual_seed(23);
    const kael::Tensor x = kael::randn({6, 8}).slice(1, 0, 8, 2);
    EXPECT_FALSE(x.is_contiguous());
    EXPECT_TRUE(kael::test::allclose(kael::sum(kael::softmax(x, -1), {-1}), kael::ones({6}),
                                     {.tolerance = kael::test::Tolerance{1e-5, 1e-5}}));
    const kael::Tensor wide = kael::randn({2, 40000});
    EXPECT_TRUE(kael::test::allclose(kael::sum(kael::softmax(wide, -1), {-1}), kael::ones({2}),
                                     {.tolerance = kael::test::Tolerance{1e-3, 1e-3}}));
}

TEST(FusedKernels, StandardizeGradcheckOverOneAndTwoDimensions) {
    kael::manual_seed(24);
    const kael::Tensor x = kael::randn({3, 2, 4}, kael::DType::F64).requires_grad_();
    const auto one = kael::autograd::gradcheck(
        [](const std::vector<kael::Tensor>& in) { return kael::standardize(in[0], 1, 1e-5); }, {x});
    EXPECT_TRUE(one.passed) << one.report;
    const auto two = kael::autograd::gradcheck(
        [](const std::vector<kael::Tensor>& in) { return kael::standardize(in[0], 2, 1e-5); }, {x});
    EXPECT_TRUE(two.passed) << two.report;
}

TEST(FusedKernels, StandardizeRejectsBadDimensionCount) {
    EXPECT_THROW((void)kael::standardize(kael::zeros({2, 2}), 0, 1e-5), kael::Error);
    EXPECT_THROW((void)kael::standardize(kael::zeros({2, 2}), 3, 1e-5), kael::Error);
}
