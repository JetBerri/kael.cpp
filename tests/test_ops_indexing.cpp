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
using kael::Tensor;
using kael::test::values;

namespace {

Tensor index_tensor(std::vector<std::int64_t> data, const Shape& shape) {
    return kael::from_vector(data, shape);
}

} // namespace

TEST(Cat, AlongEachDim) {
    auto a = kael::arange(0, 6).reshape({2, 3});
    auto b = kael::arange(6, 9).reshape({1, 3});
    EXPECT_EQ(kael::cat({a, b}, 0).shape(), (Shape{3, 3}));
    EXPECT_EQ(values<float>(kael::cat({a, b}, 0)), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(kael::cat({a, a}, -1).shape(), (Shape{2, 6}));
}

TEST(Cat, SingleTensorAndEmptyPieces) {
    auto a = kael::ones({2});
    EXPECT_EQ(values<float>(kael::cat({a})), (std::vector<float>{1, 1}));
    EXPECT_EQ(kael::cat({kael::zeros({0, 3}), kael::zeros({2, 3})}).shape(), (Shape{2, 3}));
    EXPECT_FALSE(kael::cat({a}).shares_storage(a));
}

TEST(Cat, PromotesDtype) {
    EXPECT_EQ(kael::cat({kael::ones({1}), kael::ones({1}, DType::F64)}).dtype(), DType::F64);
}

TEST(Cat, Errors) {
    EXPECT_THROW(kael::cat({}), kael::Error);
    EXPECT_THROW(kael::cat({kael::zeros({2, 3}), kael::zeros({2, 2})}, 0), kael::Error);
    EXPECT_THROW(kael::cat({kael::zeros({2}), kael::zeros({2, 2})}), kael::Error);
    EXPECT_THROW(kael::cat({kael::zeros({2})}, 1), kael::Error);
}

TEST(Stack, AddsADimension) {
    auto a = kael::zeros({2, 3});
    EXPECT_EQ(kael::stack({a, a, a}).shape(), (Shape{3, 2, 3}));
    EXPECT_EQ(kael::stack({a, a}, 2).shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(kael::stack({a, a}, -1).shape(), (Shape{2, 3, 2}));
    EXPECT_THROW(kael::stack({a, kael::zeros({3, 2})}), kael::Error);
    EXPECT_THROW(kael::stack({}), kael::Error);
}

TEST(Where, PicksByCondition) {
    auto cond = kael::from_vector({true, false, true});
    auto out = kael::where(cond, kael::from_vector({1.f, 2.f, 3.f}), kael::from_vector({4.f, 5.f, 6.f}));
    EXPECT_EQ(values<float>(out), (std::vector<float>{1, 5, 3}));
}

TEST(Where, BroadcastsAllThree) {
    auto cond = kael::from_vector({true, false}).reshape({2, 1});
    auto out = kael::where(cond, kael::arange(0, 3), kael::scalar(9));
    EXPECT_EQ(out.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(out), (std::vector<float>{0, 1, 2, 9, 9, 9}));
}

TEST(Where, AcceptsScalars) {
    auto cond = kael::from_vector({true, false});
    EXPECT_EQ(values<float>(kael::where(cond, 1.5, kael::zeros({2}))), (std::vector<float>{1.5f, 0}));
    EXPECT_EQ(values<float>(kael::where(cond, kael::zeros({2}), 1.5)), (std::vector<float>{0, 1.5f}));
    EXPECT_EQ(kael::where(cond, 1, 2).dtype(), DType::I64);
}

TEST(Where, NanInBranchesIsKept) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto out = kael::where(kael::from_vector({false}), kael::from_vector({nan}), kael::from_vector({2.f}));
    EXPECT_EQ(out.item<float>(), 2.f);
}

TEST(Where, ConditionMustBeBool) {
    EXPECT_THROW(kael::where(kael::zeros({1}), 1, 2), kael::Error);
}

TEST(MaskedFill, CopyLeavesOriginalAlone) {
    auto t = kael::zeros({3});
    auto out = kael::masked_fill(t, kael::from_vector({true, false, true}), 5);
    EXPECT_EQ(values<float>(out), (std::vector<float>{5, 0, 5}));
    EXPECT_EQ(values<float>(t), (std::vector<float>{0, 0, 0}));
}

TEST(MaskedFill, InPlaceAndBroadcastMask) {
    auto t = kael::zeros({2, 2});
    t.masked_fill_(kael::from_vector({true, false}), 1);
    EXPECT_EQ(values<float>(t), (std::vector<float>{1, 0, 1, 0}));
}

TEST(MaskedFill, MinusInfinityForCausalMasks) {
    auto out = kael::masked_fill(kael::zeros({1}), kael::from_vector({true}),
                                 -std::numeric_limits<float>::infinity());
    EXPECT_EQ(out.item<float>(), -std::numeric_limits<float>::infinity());
}

TEST(MaskedFill, Errors) {
    auto t = kael::zeros({2});
    EXPECT_THROW(kael::masked_fill(t, kael::zeros({2}), 1), kael::Error);
    EXPECT_THROW(kael::masked_fill(t, kael::from_vector({true, false}).reshape({2, 1}), 1),
                 kael::Error);
    auto expanded = kael::zeros({2}).expand({2, 2});
    EXPECT_THROW(expanded.masked_fill_(kael::from_vector({true, true}), 1), kael::Error);
}

TEST(IndexSelect, PicksRowsAndColumns) {
    auto t = kael::arange(0, 6).reshape({2, 3});
    auto rows = kael::index_select(t, 0, index_tensor({1, 1}, {2}));
    EXPECT_EQ(values<float>(rows), (std::vector<float>{3, 4, 5, 3, 4, 5}));
    auto columns = kael::index_select(t, -1, index_tensor({2, 0}, {2}));
    EXPECT_EQ(values<float>(columns), (std::vector<float>{2, 0, 5, 3}));
}

TEST(IndexSelect, EmptyIndices) {
    auto out = kael::index_select(kael::zeros({2, 3}), 1, index_tensor({}, {0}));
    EXPECT_EQ(out.shape(), (Shape{2, 0}));
}

TEST(IndexSelect, Errors) {
    auto t = kael::zeros({2, 3});
    EXPECT_THROW(kael::index_select(t, 0, index_tensor({2}, {1})), kael::Error);
    EXPECT_THROW(kael::index_select(t, 0, index_tensor({-1}, {1})), kael::Error);
    EXPECT_THROW(kael::index_select(t, 0, index_tensor({0, 1}, {1, 2})), kael::Error);
    EXPECT_THROW(kael::index_select(t, 0, kael::zeros({1})), kael::Error);
}

TEST(Gather, GuideExample) {
    auto t = index_tensor({10, 11, 12, 20, 21, 22}, {2, 3});
    auto out = kael::gather(t, 1, index_tensor({2, 0, 1, 1}, {2, 2}));
    EXPECT_EQ(out.shape(), (Shape{2, 2}));
    EXPECT_EQ(values<std::int64_t>(out), (std::vector<std::int64_t>{12, 10, 21, 21}));
}

TEST(Gather, AlongDimZero) {
    auto t = index_tensor({10, 11, 20, 21}, {2, 2});
    auto out = kael::gather(t, 0, index_tensor({1, 0}, {1, 2}));
    EXPECT_EQ(values<std::int64_t>(out), (std::vector<std::int64_t>{20, 11}));
}

TEST(Gather, Errors) {
    auto t = kael::zeros({2, 3});
    EXPECT_THROW(kael::gather(t, 1, index_tensor({3, 0}, {2, 1})), kael::Error);
    EXPECT_THROW(kael::gather(t, 1, index_tensor({-1, 0}, {2, 1})), kael::Error);
    EXPECT_THROW(kael::gather(t, 1, index_tensor({0, 0}, {2})), kael::Error);
    EXPECT_THROW(kael::gather(t, 1, index_tensor({0, 0, 0}, {3, 1})), kael::Error);
    EXPECT_THROW(kael::gather(t, 1, kael::zeros({2, 1})), kael::Error);
}

TEST(ScatterAdd, RepeatedTargetsAccumulate) {
    auto t = kael::zeros({1, 3});
    auto source = kael::from_vector({1.f, 2.f, 4.f}, {1, 3});
    t.scatter_add_(1, index_tensor({2, 2, 0}, {1, 3}), source);
    EXPECT_EQ(values<float>(t), (std::vector<float>{4, 0, 3}));
}

TEST(ScatterAdd, ReversesGatherForGradients) {
    auto gradient = kael::zeros({2, 3});
    gradient.scatter_add_(1, index_tensor({1, 2}, {2, 1}), kael::ones({2, 1}));
    EXPECT_EQ(values<float>(gradient), (std::vector<float>{0, 1, 0, 0, 0, 1}));
}

TEST(ScatterAdd, Errors) {
    auto t = kael::zeros({2, 3});
    EXPECT_THROW(t.scatter_add_(1, index_tensor({3}, {1, 1}), kael::ones({1, 1})), kael::Error);
    EXPECT_THROW(t.scatter_add_(1, index_tensor({0, 0}, {2, 1}), kael::ones({1, 1})), kael::Error);
}
