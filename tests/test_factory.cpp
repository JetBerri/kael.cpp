#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"

#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::test::values;

namespace {

void expect_fresh_contiguous(const kael::Tensor& t) {
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.strides(), kael::contiguous_strides(t.shape()));
}

} // namespace

TEST(Factory, EmptyHasShapeAndDtype) {
    auto t = kael::empty({2, 3}, DType::F64);
    EXPECT_EQ(t.shape(), (Shape{2, 3}));
    EXPECT_EQ(t.dtype(), DType::F64);
    EXPECT_EQ(t.impl()->storage->nbytes(), 48);
    expect_fresh_contiguous(t);
}

TEST(Factory, DefaultDtypeIsFloat32) {
    EXPECT_EQ(kael::empty({1}).dtype(), DType::F32);
    EXPECT_EQ(kael::zeros({1}).dtype(), DType::F32);
    EXPECT_EQ(kael::arange(3).dtype(), DType::F32);
    EXPECT_EQ(kael::scalar(1).dtype(), DType::F32);
}

TEST(Factory, ZerosAndOnes) {
    EXPECT_EQ(values<float>(kael::zeros({2, 2})), std::vector<float>(4, 0.0f));
    EXPECT_EQ(values<std::int64_t>(kael::zeros({3}, DType::I64)), std::vector<std::int64_t>(3, 0));
    EXPECT_EQ(values<double>(kael::ones({2, 3}, DType::F64)), std::vector<double>(6, 1.0));
    EXPECT_EQ(values<std::uint8_t>(kael::ones({2}, DType::Bool)), std::vector<std::uint8_t>(2, 1));
    expect_fresh_contiguous(kael::ones({4, 5}));
}

TEST(Factory, Full) {
    EXPECT_EQ(values<float>(kael::full({3}, 2.5)), std::vector<float>(3, 2.5f));
    EXPECT_EQ(values<std::int64_t>(kael::full({2}, -7.9, DType::I64)),
              std::vector<std::int64_t>(2, -7));
}

TEST(Factory, FullBoolNonZeroIsTrue) {
    EXPECT_EQ(values<std::uint8_t>(kael::full({2}, 2.5, DType::Bool)),
              std::vector<std::uint8_t>(2, 1));
    EXPECT_EQ(values<std::uint8_t>(kael::full({2}, 0.0, DType::Bool)),
              std::vector<std::uint8_t>(2, 0));
}

TEST(Factory, LikeVariants) {
    auto base = kael::ones({2, 3}, DType::I64);
    auto e = kael::empty_like(base);
    auto z = kael::zeros_like(base);
    auto o = kael::ones_like(base);
    for (const auto& t : {e, z, o}) {
        EXPECT_EQ(t.shape(), base.shape());
        EXPECT_EQ(t.dtype(), DType::I64);
        EXPECT_FALSE(t.shares_storage(base));
    }
    EXPECT_EQ(values<std::int64_t>(z), std::vector<std::int64_t>(6, 0));
    EXPECT_EQ(values<std::int64_t>(o), std::vector<std::int64_t>(6, 1));
}

TEST(Factory, ArangeExamples) {
    EXPECT_EQ(values<float>(kael::arange(0, 5, 1)), (std::vector<float>{0, 1, 2, 3, 4}));
    EXPECT_EQ(values<float>(kael::arange(5)), (std::vector<float>{0, 1, 2, 3, 4}));
    auto t = kael::arange(0, 1, 0.3, DType::F64);
    ASSERT_EQ(t.numel(), 4);
    EXPECT_DOUBLE_EQ(t.data_ptr<double>()[3], 0.9);
    EXPECT_EQ(values<std::int64_t>(kael::arange(5, 0, -2, DType::I64)),
              (std::vector<std::int64_t>{5, 3, 1}));
    expect_fresh_contiguous(t);
}

TEST(Factory, ArangeIntegersTruncateStartAndStepLikePyTorch) {
    EXPECT_EQ(values<std::int64_t>(kael::arange(0.5, 3.5, 1.5, DType::I64)),
              (std::vector<std::int64_t>{0, 1}));
    EXPECT_EQ(values<std::int64_t>(kael::arange(-0.5, 3.0, 1.5, DType::I64)),
              (std::vector<std::int64_t>{0, 1, 2}));
}

TEST(Factory, FromVectorNormalizesBoolBytes) {
    auto t = kael::from_vector(std::vector<std::uint8_t>{2, 0});
    EXPECT_EQ(values<std::uint8_t>(t), (std::vector<std::uint8_t>{1, 0}));
}

TEST(Factory, ArangeEmptyRange) {
    EXPECT_EQ(kael::arange(3, 3, 1).numel(), 0);
}

TEST(Factory, ArangeErrors) {
    EXPECT_THROW(kael::arange(0, 5, 0), kael::Error);
    EXPECT_THROW(kael::arange(0, 5, -1), kael::Error);
    EXPECT_THROW(kael::arange(5, 0, 1), kael::Error);
    EXPECT_THROW(kael::arange(0, 5, 1, DType::Bool), kael::Error);
}

TEST(Factory, Linspace) {
    EXPECT_EQ(kael::linspace(0, 1, 0).numel(), 0);
    EXPECT_EQ(values<float>(kael::linspace(3, 7, 1)), (std::vector<float>{3}));
    EXPECT_EQ(values<double>(kael::linspace(0, 1, 5, DType::F64)),
              (std::vector<double>{0, 0.25, 0.5, 0.75, 1.0}));
}

TEST(Factory, LinspaceLastIsExactlyEnd) {
    auto t = kael::linspace(0.1, 0.7, 7, DType::F64);
    EXPECT_EQ(t.data_ptr<double>()[6], 0.7);
    auto f = kael::linspace(-1.3, 2.9, 13);
    EXPECT_EQ(f.data_ptr<float>()[12], 2.9f);
}

TEST(Factory, LinspaceErrors) {
    EXPECT_THROW(kael::linspace(0, 1, -1), kael::Error);
    EXPECT_THROW(kael::linspace(0, 1, 5, DType::I64), kael::Error);
}

TEST(Factory, Eye) {
    EXPECT_EQ(values<float>(kael::eye(3)), (std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1}));
    EXPECT_EQ(values<std::uint8_t>(kael::eye(2, DType::Bool)),
              (std::vector<std::uint8_t>{1, 0, 0, 1}));
    EXPECT_EQ(kael::eye(0).shape(), (Shape{0, 0}));
}

TEST(Factory, Scalar) {
    auto s = kael::scalar(3.5, DType::F64);
    EXPECT_EQ(s.dim(), 0);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_EQ(s.data_ptr<double>()[0], 3.5);
}

TEST(Factory, FromVector) {
    auto t = kael::from_vector(std::vector<double>{1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_EQ(t.dtype(), DType::F64);
    EXPECT_EQ(t.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<double>(t), (std::vector<double>{1, 2, 3, 4, 5, 6}));
    expect_fresh_contiguous(t);
}

TEST(Factory, FromVectorLiteralsDecideDtype) {
    EXPECT_EQ(kael::from_vector({1.f, 2.f, 3.f}, {3}).dtype(), DType::F32);
    EXPECT_EQ(kael::from_vector({1.0, 2.0}).dtype(), DType::F64);
    EXPECT_EQ(kael::from_vector({std::int64_t{1}, std::int64_t{2}}).dtype(), DType::I64);
}

TEST(Factory, FromVectorOneDimensional) {
    auto t = kael::from_vector(std::vector<std::int64_t>{4, 5, 6});
    EXPECT_EQ(t.shape(), (Shape{3}));
    EXPECT_EQ(values<std::int64_t>(t), (std::vector<std::int64_t>{4, 5, 6}));
}

TEST(Factory, FromVectorBool) {
    auto t = kael::from_vector(std::vector<bool>{true, false, true}, {3});
    EXPECT_EQ(t.dtype(), DType::Bool);
    EXPECT_EQ(values<std::uint8_t>(t), (std::vector<std::uint8_t>{1, 0, 1}));
    EXPECT_EQ(kael::from_vector({true, false}).dtype(), DType::Bool);
}

TEST(Factory, FromVectorWrongSizeThrows) {
    EXPECT_THROW(kael::from_vector(std::vector<float>{1, 2, 3}, {2, 2}), kael::Error);
    EXPECT_THROW(kael::from_vector(std::vector<bool>{true}, {2}), kael::Error);
}

TEST(Factory, ZeroSizedDimensionAllocatesNothing) {
    auto t = kael::zeros({3, 0, 2});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.impl()->storage->nbytes(), 0);
    EXPECT_EQ(t.impl()->storage->data(), nullptr);
    EXPECT_EQ(kael::from_vector(std::vector<float>{}, {0}).numel(), 0);
}

TEST(Factory, NegativeShapeThrows) {
    EXPECT_THROW(kael::zeros({2, -1}), kael::Error);
}

TEST(Factory, BytesOverflowThrows) {
    EXPECT_THROW(kael::empty({INT64_MAX / 4, 2}, DType::F64), kael::Error);
}
