#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::test::values;

namespace {

const float kNan = std::numeric_limits<float>::quiet_NaN();

kael::Tensor iota_2x3() {
    return kael::arange(0, 6).reshape({2, 3});
}

} // namespace

TEST(Reduce, SumAllAndAlongDims) {
    auto t = iota_2x3();
    EXPECT_EQ(kael::sum(t).item<float>(), 15.f);
    EXPECT_EQ(kael::sum(t).dim(), 0);
    EXPECT_EQ(values<float>(kael::sum(t, {0})), (std::vector<float>{3, 5, 7}));
    EXPECT_EQ(values<float>(kael::sum(t, {-1})), (std::vector<float>{3, 12}));
    EXPECT_EQ(kael::sum(t, {0, 1}).item<float>(), 15.f);
}

TEST(Reduce, KeepdimLeavesSizeOne) {
    auto t = kael::zeros({2, 3, 4});
    EXPECT_EQ(kael::sum(t, {1}).shape(), (Shape{2, 4}));
    EXPECT_EQ(kael::sum(t, {1}, true).shape(), (Shape{2, 1, 4}));
    EXPECT_EQ(kael::sum(t, {}, true).shape(), (Shape{1, 1, 1}));
}

TEST(Reduce, RepeatedOrInvalidDimsThrow) {
    auto t = iota_2x3();
    EXPECT_THROW(kael::sum(t, {0, 0}), kael::Error);
    EXPECT_THROW(kael::sum(t, {0, -2}), kael::Error);
    EXPECT_THROW(kael::sum(t, {2}), kael::Error);
}

TEST(Reduce, ScalarsAcceptDimZero) {
    auto s = kael::scalar(3);
    EXPECT_EQ(kael::sum(s).item<float>(), 3.f);
    EXPECT_EQ(kael::sum(s, {0}).item<float>(), 3.f);
    EXPECT_EQ(kael::sum(s, {-1}).item<float>(), 3.f);
    EXPECT_THROW(kael::sum(s, {1}), kael::Error);
}

TEST(Reduce, DtypeOfSum) {
    EXPECT_EQ(kael::sum(kael::ones({2}, DType::Bool)).dtype(), DType::I64);
    EXPECT_EQ(kael::sum(kael::ones({2}, DType::Bool)).item<std::int64_t>(), 2);
    EXPECT_EQ(kael::sum(kael::ones({2}, DType::I64)).dtype(), DType::I64);
    EXPECT_EQ(kael::sum(kael::ones({2}, DType::F64)).dtype(), DType::F64);
    EXPECT_EQ(kael::prod(kael::full({2}, 3, DType::Bool)).dtype(), DType::I64);
}

TEST(Reduce, EmptyInputs) {
    EXPECT_EQ(kael::sum(kael::zeros({0})).item<float>(), 0.f);
    EXPECT_EQ(kael::prod(kael::zeros({0})).item<float>(), 1.f);
    EXPECT_TRUE(std::isnan(kael::mean(kael::zeros({0})).item<float>()));
    EXPECT_THROW(kael::max(kael::zeros({0})), kael::Error);
    EXPECT_THROW(kael::min(kael::zeros({0})), kael::Error);
    EXPECT_THROW(kael::argmax(kael::zeros({0})), kael::Error);
    EXPECT_EQ(kael::max(kael::zeros({0, 3}), {1}).shape(), (Shape{0}));
    EXPECT_THROW(kael::max(kael::zeros({0, 3}), {0}), kael::Error);
    EXPECT_EQ(kael::all(kael::zeros({0})).item<std::uint8_t>(), 1);
    EXPECT_EQ(kael::any(kael::zeros({0})).item<std::uint8_t>(), 0);
}

TEST(Reduce, SumOfManyFloatsDoesNotDrift) {
    auto t = kael::full({10000000}, 0.1);
    EXPECT_NEAR(kael::sum(t).item<float>(), 1000000.f, 1.f);
}

TEST(Reduce, MeanOnlyFloats) {
    EXPECT_EQ(values<float>(kael::mean(iota_2x3(), {0})), (std::vector<float>{1.5f, 2.5f, 3.5f}));
    EXPECT_THROW(kael::mean(kael::ones({2}, DType::I64)), kael::Error);
    EXPECT_THROW(kael::mean(kael::ones({2}, DType::Bool)), kael::Error);
}

TEST(Reduce, MaxAndMinPropagateNan) {
    auto t = kael::from_vector({1.f, kNan, 3.f});
    EXPECT_TRUE(std::isnan(kael::max(t).item<float>()));
    EXPECT_TRUE(std::isnan(kael::min(t).item<float>()));
    EXPECT_EQ(kael::max(iota_2x3()).item<float>(), 5.f);
    EXPECT_EQ(values<float>(kael::min(iota_2x3(), {1})), (std::vector<float>{0, 3}));
}

TEST(Reduce, ArgmaxTiesTakeTheFirst) {
    auto t = kael::from_vector({1.f, 3.f, 3.f, 0.f, 0.f});
    EXPECT_EQ(kael::argmax(t).item<std::int64_t>(), 1);
    EXPECT_EQ(kael::argmin(t).item<std::int64_t>(), 3);
    EXPECT_EQ(kael::argmax(t).dtype(), DType::I64);
}

TEST(Reduce, ArgmaxAlongDim) {
    auto t = kael::from_vector({1.f, 5.f, 2.f, 9.f, 0.f, 3.f}, {2, 3});
    EXPECT_EQ(values<std::int64_t>(kael::argmax(t, 1)), (std::vector<std::int64_t>{1, 0}));
    EXPECT_EQ(kael::argmax(t, 1, true).shape(), (Shape{2, 1}));
    EXPECT_EQ(kael::argmax(t).item<std::int64_t>(), 3);
    EXPECT_EQ(kael::argmin(t).item<std::int64_t>(), 4);
}

TEST(Reduce, VarExample) {
    auto x = kael::from_vector({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
    EXPECT_NEAR(kael::var(x).item<double>(), 32.0 / 7.0, 1e-12);
    EXPECT_NEAR(kael::var(x, {}, false, 0).item<double>(), 4.0, 1e-12);
    EXPECT_NEAR(kael::std(x).item<double>(), std::sqrt(32.0 / 7.0), 1e-12);
}

TEST(Reduce, VarWithTooFewElementsIsNan) {
    EXPECT_TRUE(std::isnan(kael::var(kael::ones({1})).item<float>()));
    EXPECT_THROW(kael::var(kael::ones({2}, DType::I64)), kael::Error);
}

TEST(Reduce, VarDoesNotLosePrecisionWithLargeMean) {
    auto x = kael::from_vector({1e8 + 1.0, 1e8 - 1.0});
    EXPECT_NEAR(kael::var(x, {}, false, 0).item<double>(), 1.0, 1e-9);
}

TEST(Reduce, AllAndAny) {
    auto t = kael::from_vector({0.f, 2.f, kNan, 1.f}, {2, 2});
    EXPECT_EQ(kael::all(t).item<std::uint8_t>(), 0);
    EXPECT_EQ(kael::any(t).item<std::uint8_t>(), 1);
    EXPECT_EQ(values<std::uint8_t>(kael::all(t, {1})), (std::vector<std::uint8_t>{0, 1}));
    EXPECT_EQ(kael::any(t).dtype(), DType::Bool);
}

TEST(Reduce, NormIsEuclidean) {
    EXPECT_EQ(kael::norm(kael::from_vector({3.f, 4.f})).item<float>(), 5.f);
    EXPECT_THROW(kael::norm(kael::ones({2}, DType::I64)), kael::Error);
}

TEST(Reduce, Methods) {
    auto t = iota_2x3();
    EXPECT_EQ(t.sum().item<float>(), 15.f);
    EXPECT_EQ(values<float>(t.mean({1})), (std::vector<float>{1, 4}));
}

TEST(Reduce, NonContiguousInputs) {
    auto t = iota_2x3().T();
    EXPECT_EQ(values<float>(kael::sum(t, {0})), (std::vector<float>{3, 12}));
    auto expanded = kael::arange(0, 3).expand({2, 3});
    EXPECT_EQ(values<float>(kael::sum(expanded, {0})), (std::vector<float>{0, 2, 4}));
}
