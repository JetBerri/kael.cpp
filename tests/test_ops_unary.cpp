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
using kael::test::values;

namespace {

const float kInf = std::numeric_limits<float>::infinity();
const float kNan = std::numeric_limits<float>::quiet_NaN();

float single(const kael::Tensor& t) {
    return t.item<float>();
}

kael::Tensor one(float value) {
    return kael::scalar(value);
}

kael::Tensor int_tensor() {
    return kael::from_vector(std::vector<std::int64_t>{-2, 0, 3});
}

} // namespace

TEST(Unary, NegOfBoolThrows) {
    EXPECT_THROW(kael::neg(kael::from_vector({true})), kael::Error);
    EXPECT_EQ(values<float>(-kael::from_vector({1.f, -2.f})), (std::vector<float>{-1, 2}));
}

TEST(Unary, KeepDtype) {
    EXPECT_EQ(kael::abs(int_tensor()).dtype(), DType::I64);
    EXPECT_EQ(kael::neg(int_tensor()).dtype(), DType::I64);
    EXPECT_EQ(kael::sign(int_tensor()).dtype(), DType::I64);
    EXPECT_EQ(kael::relu(int_tensor()).dtype(), DType::I64);
    EXPECT_EQ(values<std::int64_t>(kael::abs(int_tensor())), (std::vector<std::int64_t>{2, 0, 3}));
    EXPECT_EQ(values<std::int64_t>(kael::sign(int_tensor())),
              (std::vector<std::int64_t>{-1, 0, 1}));
    EXPECT_EQ(values<std::int64_t>(kael::relu(int_tensor())), (std::vector<std::int64_t>{0, 0, 3}));
}

TEST(Unary, MathOnIntegersIsFloat32) {
    for (auto op : {kael::exp, kael::log, kael::sqrt, kael::sin, kael::tanh, kael::sigmoid}) {
        EXPECT_EQ(op(int_tensor()).dtype(), DType::F32);
        EXPECT_EQ(op(kael::from_vector({true})).dtype(), DType::F32);
    }
    EXPECT_EQ(kael::exp(kael::zeros({1}, DType::F64)).dtype(), DType::F64);
}

TEST(Unary, LogSpecialValues) {
    EXPECT_EQ(single(kael::log(one(0))), -kInf);
    EXPECT_TRUE(std::isnan(single(kael::log(one(-1)))));
}

TEST(Unary, SqrtOfNegativeIsNan) {
    EXPECT_TRUE(std::isnan(single(kael::sqrt(one(-1)))));
    EXPECT_EQ(single(kael::rsqrt(one(4))), 0.5f);
}

TEST(Unary, Log1pAndExpm1KeepPrecisionNearZero) {
    const auto tiny = kael::scalar(1e-10, DType::F64);
    EXPECT_NEAR(kael::log1p(tiny).item<double>(), 1e-10, 1e-19);
    EXPECT_NEAR(kael::expm1(tiny).item<double>(), 1e-10, 1e-19);
}

TEST(Unary, SigmoidIsStableAtExtremes) {
    EXPECT_EQ(single(kael::sigmoid(one(-1000))), 0.f);
    EXPECT_EQ(single(kael::sigmoid(one(1000))), 1.f);
    EXPECT_EQ(single(kael::sigmoid(one(0))), 0.5f);
    EXPECT_TRUE(std::isnan(single(kael::sigmoid(one(kNan)))));
}

TEST(Unary, ReluPropagatesNan) {
    EXPECT_TRUE(std::isnan(single(kael::relu(one(kNan)))));
    EXPECT_EQ(single(kael::relu(one(-3))), 0.f);
    EXPECT_EQ(single(kael::relu(one(3))), 3.f);
}

TEST(Unary, SignOfNanIsZeroLikePyTorch) {
    EXPECT_EQ(single(kael::sign(one(kNan))), 0.f);
}

TEST(Clamp, LimitsBothSides) {
    auto out = kael::clamp(kael::from_vector({-2.f, 0.f, 2.f, kNan}), -1, 1);
    auto v = values<float>(out);
    EXPECT_EQ(v[0], -1.f);
    EXPECT_EQ(v[1], 0.f);
    EXPECT_EQ(v[2], 1.f);
    EXPECT_TRUE(std::isnan(v[3]));
}

TEST(Clamp, ScalarCategoryPromotes) {
    EXPECT_EQ(kael::clamp(int_tensor(), -1, 1).dtype(), DType::I64);
    EXPECT_EQ(kael::clamp(int_tensor(), -0.5, 0.5).dtype(), DType::F32);
    EXPECT_EQ(values<float>(kael::clamp(int_tensor(), -0.5, 0.5)),
              (std::vector<float>{-0.5f, 0.f, 0.5f}));
}

TEST(Unary, EmptyAndNonContiguous) {
    EXPECT_EQ(kael::exp(kael::zeros({0, 2})).numel(), 0);
    auto t = kael::arange(0, 4).reshape({2, 2}).T();
    EXPECT_EQ(values<float>(kael::abs(t)), (std::vector<float>{0, 2, 1, 3}));
}
