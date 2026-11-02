#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/tensor_impl.h"

namespace {

// An impl over a fresh storage of `nbytes` bytes.
kael::TensorImpl make_impl(std::int64_t nbytes, kael::Shape shape, kael::Strides strides,
                           std::int64_t offset = 0, kael::DType dtype = kael::DType::F32) {
    kael::TensorImpl impl;
    impl.storage = std::make_shared<kael::Storage>(nbytes);
    impl.shape = std::move(shape);
    impl.strides = std::move(strides);
    impl.offset = offset;
    impl.dtype = dtype;
    return impl;
}

} // namespace

TEST(TensorImpl, NumelDimContiguous) {
    auto impl = make_impl(24, {2, 3}, {3, 1});
    EXPECT_EQ(impl.numel(), 6);
    EXPECT_EQ(impl.dim(), 2);
    EXPECT_TRUE(impl.is_contiguous());
    EXPECT_NO_THROW(impl.validate());
}

TEST(TensorImpl, TransposedStridesAreNotContiguous) {
    auto impl = make_impl(24, {2, 3}, {1, 2});
    EXPECT_FALSE(impl.is_contiguous());
    EXPECT_NO_THROW(impl.validate());
}

TEST(TensorImpl, ScalarImpl) {
    auto impl = make_impl(8, {}, {}, 0, kael::DType::F64);
    EXPECT_EQ(impl.numel(), 1);
    EXPECT_EQ(impl.dim(), 0);
    EXPECT_TRUE(impl.is_contiguous());
    EXPECT_NO_THROW(impl.validate());
}

TEST(TensorImplValidate, ExactlyAtTheLimit) {
    // [2, 3] float32 contiguous: max offset 5, needs (5 + 1) * 4 = 24 bytes.
    EXPECT_NO_THROW(make_impl(24, {2, 3}, {3, 1}).validate());
    // Slice starting at element 4 of 10 floats, shape [3], stride [2]:
    // max offset 8, needs 36 bytes; 40 available.
    EXPECT_NO_THROW(make_impl(40, {3}, {2}, 4).validate());
    EXPECT_NO_THROW(make_impl(36, {3}, {2}, 4).validate());
}

TEST(TensorImplValidate, OneByteShortThrows) {
    EXPECT_THROW(make_impl(23, {2, 3}, {3, 1}).validate(), kael::Error);
    EXPECT_THROW(make_impl(35, {3}, {2}, 4).validate(), kael::Error);
}

TEST(TensorImplValidate, OffsetPastTheEndThrows) {
    EXPECT_THROW(make_impl(24, {2, 3}, {3, 1}, 1).validate(), kael::Error);
}

TEST(TensorImplValidate, NegativeStrideThrows) {
    EXPECT_THROW(make_impl(24, {2, 3}, {3, -1}, 2).validate(), kael::Error);
}

TEST(TensorImplValidate, NegativeOffsetThrows) {
    EXPECT_THROW(make_impl(24, {2, 3}, {3, 1}, -1).validate(), kael::Error);
}

TEST(TensorImplValidate, StridesSizeMismatchThrows) {
    EXPECT_THROW(make_impl(24, {2, 3}, {1}).validate(), kael::Error);
}

TEST(TensorImplValidate, NegativeShapeThrows) {
    EXPECT_THROW(make_impl(24, {2, -3}, {3, 1}).validate(), kael::Error);
}

TEST(TensorImplValidate, NullStorageThrows) {
    kael::TensorImpl impl;
    impl.shape = {2};
    impl.strides = {1};
    EXPECT_THROW(impl.validate(), kael::Error);
}

TEST(TensorImplValidate, EmptyTensorNeedsNoBytes) {
    EXPECT_NO_THROW(make_impl(0, {0, 3}, {3, 1}).validate());
    EXPECT_NO_THROW(make_impl(0, {3, 0}, {1, 1}, 7).validate());
}

TEST(TensorImplValidate, DtypeSizeMatters) {
    // 6 float64 elements need 48 bytes.
    EXPECT_THROW(make_impl(24, {2, 3}, {3, 1}, 0, kael::DType::F64).validate(), kael::Error);
    EXPECT_NO_THROW(make_impl(48, {2, 3}, {3, 1}, 0, kael::DType::F64).validate());
}

TEST(TensorImplValidate, HugeStridesOverflowThrows) {
    EXPECT_THROW(make_impl(64, {2, 2}, {INT64_MAX / 2, 1}).validate(), kael::Error);
}
