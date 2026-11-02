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

const float kInf = std::numeric_limits<float>::infinity();
const float kNan = std::numeric_limits<float>::quiet_NaN();

auto ints(std::vector<std::int64_t> data) {
    return kael::from_vector(data);
}

auto bools(std::vector<bool> data) {
    return kael::from_vector(data);
}

std::vector<std::uint8_t> flags(const kael::Tensor& t) {
    EXPECT_EQ(t.dtype(), DType::Bool);
    return values<std::uint8_t>(t);
}

} // namespace

TEST(Arithmetic, AddBroadcasts) {
    auto out = kael::add(kael::arange(0, 6).reshape({2, 3}), kael::arange(0, 3));
    EXPECT_EQ(out.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(out), (std::vector<float>{0, 2, 4, 3, 5, 7}));
}

TEST(Arithmetic, BoolPlusBoolIsInt64) {
    auto out = kael::add(bools({true, true}), bools({true, false}));
    EXPECT_EQ(out.dtype(), DType::I64);
    EXPECT_EQ(values<std::int64_t>(out), (std::vector<std::int64_t>{2, 1}));
}

TEST(Arithmetic, SubWithBoolThrows) {
    EXPECT_THROW(kael::sub(bools({true}), bools({false})), kael::Error);
    EXPECT_THROW(kael::sub(ints({1}), bools({false})), kael::Error);
}

TEST(Arithmetic, MulOfBoolsIsBool) {
    auto out = kael::mul(bools({true, true}), bools({true, false}));
    EXPECT_EQ(flags(out), (std::vector<std::uint8_t>{1, 0}));
}

TEST(Arithmetic, IntegerDivisionIsFloat32) {
    auto out = kael::div(ints({1, 3}), ints({2, 2}));
    EXPECT_EQ(out.dtype(), DType::F32);
    EXPECT_EQ(values<float>(out), (std::vector<float>{0.5f, 1.5f}));
}

TEST(Arithmetic, FloatDivisionByZeroFollowsIeee) {
    auto out = kael::div(kael::from_vector({1.f, -1.f, 0.f}), kael::zeros({3}));
    auto v = values<float>(out);
    EXPECT_EQ(v[0], kInf);
    EXPECT_EQ(v[1], -kInf);
    EXPECT_TRUE(std::isnan(v[2]));
}

TEST(Arithmetic, FloorDivideRoundsDown) {
    EXPECT_EQ(values<std::int64_t>(kael::floor_divide(ints({7, -7, 7, -7}), ints({2, 2, -2, -2}))),
              (std::vector<std::int64_t>{3, -4, -4, 3}));
    EXPECT_EQ(values<float>(kael::floor_divide(kael::from_vector({-7.f}), 2)),
              (std::vector<float>{-4.f}));
}

TEST(Arithmetic, FloorDivideIntegerByZeroThrows) {
    EXPECT_THROW(kael::floor_divide(ints({1}), ints({0})), kael::Error);
}

TEST(Arithmetic, PowIntegersAreExact) {
    EXPECT_EQ(values<std::int64_t>(kael::pow(ints({3, 2, -2}), ints({30, 0, 3}))),
              (std::vector<std::int64_t>{205891132094649, 1, -8}));
    EXPECT_THROW(kael::pow(ints({2}), ints({-1})), kael::Error);
}

TEST(Arithmetic, PowFloats) {
    EXPECT_EQ(values<float>(kael::pow(kael::from_vector({4.f}), 0.5f)), (std::vector<float>{2.f}));
}

TEST(Arithmetic, MaximumAndMinimumPropagateNan) {
    auto a = kael::from_vector({kNan, 1.f, 3.f});
    auto b = kael::from_vector({1.f, kNan, 2.f});
    auto highest = values<float>(kael::maximum(a, b));
    EXPECT_TRUE(std::isnan(highest[0]));
    EXPECT_TRUE(std::isnan(highest[1]));
    EXPECT_EQ(highest[2], 3.f);
    auto lowest = values<float>(kael::minimum(a, b));
    EXPECT_TRUE(std::isnan(lowest[0]));
    EXPECT_TRUE(std::isnan(lowest[1]));
    EXPECT_EQ(lowest[2], 2.f);
}

TEST(Promotion, TensorPairs) {
    EXPECT_EQ(kael::add(kael::zeros({1}), kael::zeros({1}, DType::I64)).dtype(), DType::F32);
    EXPECT_EQ(kael::add(kael::zeros({1}), kael::zeros({1}, DType::F64)).dtype(), DType::F64);
    EXPECT_EQ(kael::mul(bools({true}), ints({2})).dtype(), DType::I64);
}

TEST(Promotion, ScalarsDoNotRaisePrecision) {
    EXPECT_EQ(kael::add(kael::zeros({1}), 2.5).dtype(), DType::F32);
    EXPECT_EQ(kael::add(kael::zeros({1}, DType::F64), 2.5f).dtype(), DType::F64);
    EXPECT_EQ(kael::add(ints({1}), 2.5).dtype(), DType::F32);
    EXPECT_EQ(kael::add(ints({1}), 2).dtype(), DType::I64);
    EXPECT_EQ(kael::add(bools({true}), 2).dtype(), DType::I64);
}

TEST(Scalars, WorkOnBothSides) {
    auto t = kael::from_vector({1.f, 2.f});
    EXPECT_EQ(values<float>(kael::sub(t, 1)), (std::vector<float>{0, 1}));
    EXPECT_EQ(values<float>(kael::sub(1, t)), (std::vector<float>{0, -1}));
    EXPECT_EQ(values<float>(kael::div(2.0, t)), (std::vector<float>{2, 1}));
}

TEST(Comparisons, ReturnBoolAndCompareInPromotedType) {
    auto a = kael::from_vector({1.5f, 2.f});
    auto b = ints({2, 2});
    EXPECT_EQ(flags(kael::eq(a, b)), (std::vector<std::uint8_t>{0, 1}));
    EXPECT_EQ(flags(kael::ne(a, b)), (std::vector<std::uint8_t>{1, 0}));
    EXPECT_EQ(flags(kael::lt(a, b)), (std::vector<std::uint8_t>{1, 0}));
    EXPECT_EQ(flags(kael::le(a, b)), (std::vector<std::uint8_t>{1, 1}));
    EXPECT_EQ(flags(kael::gt(a, b)), (std::vector<std::uint8_t>{0, 0}));
    EXPECT_EQ(flags(kael::ge(a, b)), (std::vector<std::uint8_t>{0, 1}));
}

TEST(Comparisons, NanIsOnlyNotEqual) {
    auto nan = kael::from_vector({kNan});
    EXPECT_EQ(flags(kael::eq(nan, nan)), (std::vector<std::uint8_t>{0}));
    EXPECT_EQ(flags(kael::ne(nan, nan)), (std::vector<std::uint8_t>{1}));
    EXPECT_EQ(flags(kael::lt(nan, 1)), (std::vector<std::uint8_t>{0}));
    EXPECT_EQ(flags(kael::ge(nan, 1)), (std::vector<std::uint8_t>{0}));
}

TEST(Logical, TreatNonZeroAsTrue) {
    auto a = kael::from_vector({0.f, 2.5f, 0.f, kNan});
    auto b = ints({0, 0, 3, 1});
    EXPECT_EQ(flags(kael::logical_and(a, b)), (std::vector<std::uint8_t>{0, 0, 0, 1}));
    EXPECT_EQ(flags(kael::logical_or(a, b)), (std::vector<std::uint8_t>{0, 1, 1, 1}));
    EXPECT_EQ(flags(kael::logical_xor(a, b)), (std::vector<std::uint8_t>{0, 1, 1, 0}));
    EXPECT_EQ(flags(kael::logical_not(a)), (std::vector<std::uint8_t>{1, 0, 1, 0}));
}

TEST(Operators, ArithmeticAndScalars) {
    auto a = kael::from_vector({2.f, 4.f});
    auto b = kael::from_vector({1.f, 2.f});
    EXPECT_EQ(values<float>(a + b), (std::vector<float>{3, 6}));
    EXPECT_EQ(values<float>(a - b), (std::vector<float>{1, 2}));
    EXPECT_EQ(values<float>(a * b), (std::vector<float>{2, 8}));
    EXPECT_EQ(values<float>(a / b), (std::vector<float>{2, 2}));
    EXPECT_EQ(values<float>(a + 1), (std::vector<float>{3, 5}));
    EXPECT_EQ(values<float>(1 + a), (std::vector<float>{3, 5}));
    EXPECT_EQ(values<float>(2.0 * a), (std::vector<float>{4, 8}));
    EXPECT_EQ(values<float>(8 / a), (std::vector<float>{4, 2}));
}

TEST(Operators, Comparisons) {
    auto a = kael::from_vector({1.f, 2.f});
    auto b = kael::from_vector({1.f, 3.f});
    EXPECT_EQ(flags(a == b), (std::vector<std::uint8_t>{1, 0}));
    EXPECT_EQ(flags(a != b), (std::vector<std::uint8_t>{0, 1}));
    EXPECT_EQ(flags(a < b), (std::vector<std::uint8_t>{0, 1}));
    EXPECT_EQ(flags(a <= 1), (std::vector<std::uint8_t>{1, 0}));
    EXPECT_EQ(flags(a > 1), (std::vector<std::uint8_t>{0, 1}));
    EXPECT_EQ(flags(a >= b), (std::vector<std::uint8_t>{1, 0}));
}

TEST(Equal, ReturnsPlainBool) {
    auto a = kael::from_vector({1.f, 2.f});
    EXPECT_TRUE(kael::equal(a, a.clone()));
    EXPECT_FALSE(kael::equal(a, kael::from_vector({1.f, 3.f})));
    EXPECT_FALSE(kael::equal(a, kael::zeros({3})));
}

TEST(Binary, EmptyAndScalarShapes) {
    EXPECT_EQ((kael::zeros({0, 2}) + kael::zeros({2})).shape(), (Shape{0, 2}));
    EXPECT_EQ((kael::scalar(2) * kael::scalar(3)).item<float>(), 6.f);
    EXPECT_THROW(kael::zeros({3}) + kael::zeros({2}), kael::Error);
}
