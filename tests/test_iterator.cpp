#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/iterator.h"
#include "kael/scalar.h"

#include "support/values.h"

using kael::DType;
using kael::LoopDims;
using kael::Shape;
using kael::Strides;
using kael::Tensor;
using kael::TensorIterator;
using kael::test::values;

namespace {

std::vector<float> add_with_iterator(const Tensor& a, const Tensor& b) {
    TensorIterator iterator;
    iterator.allocate_output(DType::F32).add_input(a).add_input(b).build();
    iterator.run([](std::byte* const* pointers, const std::int64_t* strides, std::int64_t count) {
        for (std::int64_t i = 0; i < count; ++i) {
            *reinterpret_cast<float*>(pointers[0] + i * strides[0]) =
                *reinterpret_cast<const float*>(pointers[1] + i * strides[1]) +
                *reinterpret_cast<const float*>(pointers[2] + i * strides[2]);
        }
    });
    return values<float>(iterator.output());
}

} // namespace

TEST(Coalesce, FullyContiguousBecomesOneLoop) {
    const LoopDims merged = kael::coalesce_dims({{2, 3, 4}, {{12, 4, 1}}});
    EXPECT_EQ(merged.shape, (Shape{24}));
    EXPECT_EQ(merged.operand_strides[0], (Strides{1}));
}

TEST(Coalesce, BroadcastOperandStopsTheMerge) {
    const LoopDims merged = kael::coalesce_dims({{2, 3, 4}, {{12, 4, 1}, {0, 4, 1}}});
    EXPECT_EQ(merged.shape, (Shape{2, 12}));
    EXPECT_EQ(merged.operand_strides[0], (Strides{12, 1}));
    EXPECT_EQ(merged.operand_strides[1], (Strides{0, 1}));
}

TEST(Coalesce, TransposedMergesNothing) {
    const LoopDims merged = kael::coalesce_dims({{2, 3}, {{1, 2}}});
    EXPECT_EQ(merged.shape, (Shape{2, 3}));
    EXPECT_EQ(merged.operand_strides[0], (Strides{1, 2}));
}

TEST(Coalesce, SizeOneDimsDisappear) {
    const LoopDims merged = kael::coalesce_dims({{2, 1, 3, 1}, {{3, 99, 1, 7}}});
    EXPECT_EQ(merged.shape, (Shape{6}));
    EXPECT_EQ(merged.operand_strides[0], (Strides{1}));
}

TEST(Coalesce, AllSizeOneLeavesNoDims) {
    EXPECT_TRUE(kael::coalesce_dims({{1, 1}, {{1, 1}}}).shape.empty());
}

TEST(Coalesce, StrideZeroDimsMerge) {
    const LoopDims merged = kael::coalesce_dims({{2, 3}, {{0, 0}}});
    EXPECT_EQ(merged.shape, (Shape{6}));
    EXPECT_EQ(merged.operand_strides[0], (Strides{0}));
}

TEST(Iterator, ContiguousIsOneDimension) {
    TensorIterator iterator;
    iterator.allocate_output(DType::F32).add_input(kael::ones({2, 3})).build();
    EXPECT_EQ(iterator.dims().shape, (Shape{6}));
    EXPECT_EQ(iterator.dims().operand_strides[0], (Strides{4}));
    EXPECT_EQ(iterator.dims().operand_strides[1], (Strides{4}));
}

TEST(Iterator, StridesAreInBytesPerOperand) {
    TensorIterator iterator;
    iterator.allocate_output(DType::Bool).add_input(kael::ones({3}, DType::F64)).build();
    EXPECT_EQ(iterator.dims().operand_strides[0], (Strides{1}));
    EXPECT_EQ(iterator.dims().operand_strides[1], (Strides{8}));
}

TEST(Iterator, ContiguousAdd) {
    auto a = kael::arange(0, 6).reshape({2, 3});
    auto b = kael::full({2, 3}, 10.0);
    EXPECT_EQ(add_with_iterator(a, b), (std::vector<float>{10, 11, 12, 13, 14, 15}));
}

TEST(Iterator, BroadcastAdd) {
    auto a = kael::arange(0, 6).reshape({2, 3});
    auto b = kael::arange(0, 3);
    EXPECT_EQ(add_with_iterator(a, b), (std::vector<float>{0, 2, 4, 3, 5, 7}));
}

TEST(Iterator, TransposedAdd) {
    auto a = kael::arange(0, 6).reshape({2, 3}).T();
    auto b = kael::zeros({3, 2});
    EXPECT_EQ(add_with_iterator(a, b), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(Iterator, ThreeDimensionalOdometer) {
    auto a = kael::arange(0, 24).reshape({2, 3, 4}).permute({2, 0, 1});
    auto expected = values<float>(a);
    EXPECT_EQ(add_with_iterator(a, kael::zeros({4, 2, 3})), expected);
}

TEST(Iterator, ZeroSizedRunsNothing) {
    TensorIterator iterator;
    iterator.allocate_output(DType::F32).add_input(kael::zeros({2, 0, 3})).build();
    int calls = 0;
    iterator.run([&](std::byte* const*, const std::int64_t*, std::int64_t) { ++calls; });
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(iterator.output().shape(), (Shape{2, 0, 3}));
}

TEST(Iterator, ScalarsRunOneElement) {
    EXPECT_EQ(add_with_iterator(kael::scalar(2), kael::scalar(3)), (std::vector<float>{5}));
}

TEST(Iterator, ScalarWithLargeTensor) {
    EXPECT_EQ(add_with_iterator(kael::scalar(1), kael::zeros({3, 2})), std::vector<float>(6, 1.0f));
}

TEST(Iterator, GivenOutputIsWrittenInItsOrder) {
    auto output = kael::zeros({2, 2}).T();
    TensorIterator iterator;
    iterator.add_output(output).add_input(kael::arange(0, 4).reshape({2, 2})).build();
    iterator.run([](std::byte* const* pointers, const std::int64_t* strides, std::int64_t count) {
        for (std::int64_t i = 0; i < count; ++i) {
            *reinterpret_cast<float*>(pointers[0] + i * strides[0]) =
                *reinterpret_cast<const float*>(pointers[1] + i * strides[1]);
        }
    });
    EXPECT_EQ(values<float>(output), (std::vector<float>{0, 1, 2, 3}));
}

TEST(Iterator, BuildingTwiceIsSafe) {
    TensorIterator iterator;
    iterator.allocate_output(DType::F32).add_input(kael::ones({2})).build();
    iterator.build();
    int calls = 0;
    iterator.run([&](std::byte* const*, const std::int64_t*, std::int64_t) { ++calls; });
    EXPECT_EQ(calls, 1);
}

TEST(Iterator, Errors) {
    TensorIterator wrong_output;
    wrong_output.add_output(kael::zeros({3})).add_input(kael::zeros({2, 3}));
    EXPECT_THROW(wrong_output.build(), kael::Error);

    TensorIterator mixed_dtypes;
    mixed_dtypes.allocate_output(DType::F32)
        .add_input(kael::zeros({3}))
        .add_input(kael::zeros({3}, DType::F64));
    EXPECT_THROW(mixed_dtypes.build(), kael::Error);

    TensorIterator not_broadcastable;
    not_broadcastable.allocate_output(DType::F32)
        .add_input(kael::zeros({3}))
        .add_input(kael::zeros({2}));
    EXPECT_THROW(not_broadcastable.build(), kael::Error);

    TensorIterator no_input;
    no_input.allocate_output(DType::F32);
    EXPECT_THROW(no_input.build(), kael::Error);
}

TEST(Promotion, Table) {
    using kael::promote_types;
    static_assert(promote_types(DType::Bool, DType::Bool) == DType::Bool);
    static_assert(promote_types(DType::Bool, DType::I64) == DType::I64);
    static_assert(promote_types(DType::I64, DType::F32) == DType::F32);
    static_assert(promote_types(DType::F32, DType::I64) == DType::F32);
    static_assert(promote_types(DType::F32, DType::F64) == DType::F64);
    static_assert(promote_types(DType::F64, DType::Bool) == DType::F64);
}

TEST(Promotion, Scalars) {
    using kael::promote_with_scalar;
    static_assert(promote_with_scalar(DType::F32, 2.5) == DType::F32);
    static_assert(promote_with_scalar(DType::F64, 2.5f) == DType::F64);
    static_assert(promote_with_scalar(DType::I64, 2.5) == DType::F32);
    static_assert(promote_with_scalar(DType::Bool, 2.5) == DType::F32);
    static_assert(promote_with_scalar(DType::I64, 2) == DType::I64);
    static_assert(promote_with_scalar(DType::F32, 2) == DType::F32);
    static_assert(promote_with_scalar(DType::Bool, 2) == DType::I64);
    static_assert(promote_with_scalar(DType::Bool, true) == DType::Bool);
    static_assert(promote_with_scalar(DType::F64, true) == DType::F64);
}
