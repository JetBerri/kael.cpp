#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/tensor.h"

#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::Strides;
using kael::Tensor;
using kael::test::values;

namespace {

Tensor manual_view(const Tensor& base, Shape shape, Strides strides, std::int64_t offset = 0) {
    auto impl = std::make_shared<kael::TensorImpl>();
    impl->storage = base.impl()->storage;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = offset;
    impl->dtype = base.dtype();
    impl->device = base.device();
    return Tensor::from_impl(std::move(impl));
}

Tensor iota(const Shape& shape) {
    return kael::arange(0, static_cast<double>(kael::numel(shape)), 1, DType::I64);
}

Tensor iota_shaped(const Shape& shape) {
    return manual_view(iota(shape), shape, kael::contiguous_strides(shape));
}

} // namespace

TEST(IsContiguous, SizeOneDimensionsDoNotMatter) {
    auto base = kael::zeros({6});
    EXPECT_TRUE(manual_view(base, {1, 3}, {1, 1}).is_contiguous());
    EXPECT_TRUE(manual_view(base, {1, 3}, {3, 1}).is_contiguous());
    EXPECT_TRUE(manual_view(base, {3, 1}, {1, 99}).is_contiguous());
}

TEST(IsContiguous, TransposedIsNot) {
    auto base = kael::zeros({6});
    EXPECT_FALSE(manual_view(base, {2, 3}, {1, 2}).is_contiguous());
}

TEST(IsContiguous, GapsAreNot) {
    auto base = kael::zeros({20});
    EXPECT_FALSE(manual_view(base, {4, 3}, {5, 1}, 1).is_contiguous());
    EXPECT_FALSE(manual_view(base, {3}, {2}).is_contiguous());
}

TEST(IsContiguous, EmptyIsAlways) {
    auto base = kael::zeros({6});
    EXPECT_TRUE(manual_view(base, {0, 3}, {7, 5}).is_contiguous());
}

TEST(Contiguous, ContiguousReturnsSameImpl) {
    auto t = iota_shaped({2, 3});
    EXPECT_TRUE(t.contiguous().is_same(t));
}

TEST(Contiguous, TransposedValuesInLogicalOrder) {
    auto t = iota_shaped({2, 3});
    auto tr = manual_view(t, {3, 2}, {1, 3});
    auto c = tr.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_FALSE(c.shares_storage(tr));
    EXPECT_EQ(c.shape(), (Shape{3, 2}));
    const auto* p = c.data_ptr<std::int64_t>();
    EXPECT_EQ(std::vector<std::int64_t>(p, p + 6), (std::vector<std::int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(Clone, OfViewDoesNotShareAndIsContiguous) {
    auto t = iota_shaped({4});
    auto every_other = manual_view(t, {2}, {2}, 1);
    auto c = every_other.clone();
    EXPECT_FALSE(c.shares_storage(t));
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.offset(), 0);
    EXPECT_EQ(values<std::int64_t>(c), (std::vector<std::int64_t>{1, 3}));
}

TEST(Clone, OfContiguousViewWithOffset) {
    auto t = iota_shaped({5});
    auto tail = manual_view(t, {2}, {1}, 3);
    EXPECT_EQ(values<std::int64_t>(tail.clone()), (std::vector<std::int64_t>{3, 4}));
}

TEST(Clone, OfStride0View) {
    auto t = iota_shaped({3});
    auto expanded = manual_view(t, {2, 3}, {0, 1});
    EXPECT_EQ(values<std::int64_t>(expanded.clone()),
              (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));
}

TEST(To, WorksOnNonContiguous) {
    auto t = iota_shaped({2, 3});
    auto tr = manual_view(t, {3, 2}, {1, 3});
    auto f = tr.to(DType::F64);
    EXPECT_EQ(f.dtype(), DType::F64);
    EXPECT_EQ(f.at<double>({0, 1}), 3.0);
    EXPECT_EQ(f.at<double>({2, 1}), 5.0);
}

TEST(InferShape, ResolvesMinusOne) {
    EXPECT_EQ(kael::infer_shape({4, -1}, 12), (Shape{4, 3}));
    EXPECT_EQ(kael::infer_shape({-1}, 12), (Shape{12}));
    EXPECT_EQ(kael::infer_shape({2, 3}, 6), (Shape{2, 3}));
    EXPECT_EQ(kael::infer_shape({}, 1), (Shape{}));
    EXPECT_EQ(kael::infer_shape({-1, 3}, 0), (Shape{0, 3}));
}

TEST(InferShape, Errors) {
    EXPECT_THROW(kael::infer_shape({-1, -1}, 12), kael::Error);
    EXPECT_THROW(kael::infer_shape({-2, 6}, 12), kael::Error);
    EXPECT_THROW(kael::infer_shape({5, -1}, 12), kael::Error);
    EXPECT_THROW(kael::infer_shape({2, 3}, 12), kael::Error);
    EXPECT_THROW(kael::infer_shape({-1, 0}, 0), kael::Error);
}

TEST(View, ChangesShapeAndSharesStorage) {
    auto t = iota_shaped({2, 3});
    for (const Shape& s : {Shape{3, 2}, Shape{6}, Shape{1, 6}}) {
        auto v = t.view(s);
        EXPECT_EQ(v.shape(), s);
        EXPECT_EQ(v.strides(), kael::contiguous_strides(s));
        EXPECT_TRUE(v.shares_storage(t));
        EXPECT_EQ(values<std::int64_t>(v), (std::vector<std::int64_t>{0, 1, 2, 3, 4, 5}));
    }
    EXPECT_EQ(t.view({-1}).shape(), (Shape{6}));
    EXPECT_EQ(t.view({3, -1}).shape(), (Shape{3, 2}));
}

TEST(View, WritesReachTheOriginal) {
    auto t = iota_shaped({2, 3});
    t.view({6}).at<std::int64_t>({5}) = 100;
    EXPECT_EQ(t.at<std::int64_t>({1, 2}), 100);
}

TEST(View, KeepsOffset) {
    auto base = iota_shaped({6});
    auto tail = manual_view(base, {4}, {1}, 2);
    auto v = tail.view({2, 2});
    EXPECT_EQ(v.offset(), 2);
    EXPECT_EQ(values<std::int64_t>(v), (std::vector<std::int64_t>{2, 3, 4, 5}));
}

TEST(View, Scalar) {
    auto s = kael::scalar(4, DType::I64);
    EXPECT_EQ(s.view({1}).shape(), (Shape{1}));
    EXPECT_EQ(s.view({1, 1}).view({}).dim(), 0);
}

TEST(View, Errors) {
    auto t = iota_shaped({2, 3});
    EXPECT_THROW((void)t.view({4}), kael::Error);
    EXPECT_THROW((void)t.view({-1, -1}), kael::Error);
    EXPECT_THROW((void)manual_view(t, {3, 2}, {1, 3}).view({6}), kael::Error);
}

TEST(Reshape, ContiguousShares) {
    auto t = iota_shaped({2, 3});
    auto r = t.reshape({3, 2});
    EXPECT_TRUE(r.shares_storage(t));
}

TEST(Reshape, NonContiguousCopiesInLogicalOrder) {
    auto t = iota_shaped({2, 3});
    auto tr = manual_view(t, {3, 2}, {1, 3});
    auto r = tr.reshape({6});
    EXPECT_FALSE(r.shares_storage(t));
    EXPECT_EQ(values<std::int64_t>(r), (std::vector<std::int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(Transpose, SwapsShapeAndStrides) {
    auto t = iota_shaped({2, 3});
    auto tr = t.transpose(0, 1);
    EXPECT_EQ(tr.shape(), (Shape{3, 2}));
    EXPECT_EQ(tr.strides(), (Strides{1, 3}));
    EXPECT_EQ(tr.offset(), 0);
    EXPECT_TRUE(tr.shares_storage(t));
    EXPECT_EQ(values<std::int64_t>(tr), (std::vector<std::int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(Transpose, TwiceRestoresMetadata) {
    auto t = iota_shaped({2, 3, 4});
    auto back = t.transpose(0, 2).transpose(0, 2);
    EXPECT_EQ(back.shape(), t.shape());
    EXPECT_EQ(back.strides(), t.strides());
}

TEST(Transpose, NegativeDims) {
    auto t = iota_shaped({2, 3, 4});
    EXPECT_EQ(t.transpose(-1, 0).shape(), (Shape{4, 3, 2}));
    EXPECT_THROW((void)t.transpose(0, 3), kael::Error);
}

TEST(Permute, Example) {
    auto t = iota_shaped({2, 3, 4});
    auto p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape(), (Shape{4, 2, 3}));
    EXPECT_EQ(p.strides(), (Strides{1, 12, 4}));
}

TEST(Permute, IdentityKeepsMetadata) {
    auto t = iota_shaped({2, 3});
    auto p = t.permute({0, 1});
    EXPECT_EQ(p.shape(), t.shape());
    EXPECT_EQ(p.strides(), t.strides());
}

TEST(Permute, InverseRestoresOriginal) {
    auto t = iota_shaped({2, 3, 4});
    auto back = t.permute({2, 0, 1}).permute({1, 2, 0});
    EXPECT_EQ(back.shape(), t.shape());
    EXPECT_EQ(back.strides(), t.strides());
}

TEST(Permute, Errors) {
    auto t = iota_shaped({2, 3});
    EXPECT_THROW((void)t.permute({0, 0}), kael::Error);
    EXPECT_THROW((void)t.permute({0, -2}), kael::Error);
    EXPECT_THROW((void)t.permute({0}), kael::Error);
    EXPECT_THROW((void)t.permute({0, 2}), kael::Error);
}

TEST(MatrixTranspose, T) {
    auto m = iota_shaped({2, 3});
    EXPECT_EQ(m.T().shape(), (Shape{3, 2}));
    auto v = iota_shaped({3});
    EXPECT_TRUE(v.T().is_same(v));
    auto s = kael::scalar(1);
    EXPECT_TRUE(s.T().is_same(s));
    EXPECT_THROW((void)iota_shaped({2, 3, 4}).T(), kael::Error);
}

TEST(MatrixTranspose, MT) {
    EXPECT_EQ(iota_shaped({5, 2, 3}).mT().shape(), (Shape{5, 3, 2}));
    EXPECT_THROW((void)iota_shaped({3}).mT(), kael::Error);
}

TEST(Squeeze, DimOfSizeOneIsRemoved) {
    auto t = iota_shaped({2, 1, 3});
    auto s = t.squeeze(1);
    EXPECT_EQ(s.shape(), (Shape{2, 3}));
    EXPECT_EQ(s.strides(), (Strides{3, 1}));
    EXPECT_TRUE(s.shares_storage(t));
}

TEST(Squeeze, DimOfOtherSizeIsUnchanged) {
    auto t = iota_shaped({2, 3});
    EXPECT_TRUE(t.squeeze(0).is_same(t));
}

TEST(Squeeze, AllSizeOneDims) {
    EXPECT_EQ(iota_shaped({1, 2, 1, 3, 1}).squeeze().shape(), (Shape{2, 3}));
    EXPECT_EQ(iota_shaped({1, 1}).squeeze().dim(), 0);
}

TEST(Unsqueeze, Example) {
    auto u = iota_shaped({2, 3}).unsqueeze(1);
    EXPECT_EQ(u.shape(), (Shape{2, 1, 3}));
    EXPECT_EQ(u.strides(), (Strides{3, 3, 1}));
    EXPECT_TRUE(u.is_contiguous());
}

TEST(Unsqueeze, AtTheEnds) {
    auto t = iota_shaped({2, 3});
    EXPECT_EQ(t.unsqueeze(0).shape(), (Shape{1, 2, 3}));
    EXPECT_EQ(t.unsqueeze(2).shape(), (Shape{2, 3, 1}));
    EXPECT_EQ(t.unsqueeze(-1).shape(), (Shape{2, 3, 1}));
    EXPECT_THROW((void)t.unsqueeze(3), kael::Error);
    EXPECT_EQ(kael::scalar(1).unsqueeze(0).shape(), (Shape{1}));
}

TEST(Unsqueeze, SqueezeRecoversOriginal) {
    auto t = iota_shaped({2, 3});
    for (std::int64_t d : {0, 1, 2}) {
        auto round_trip = t.unsqueeze(d).squeeze(d);
        EXPECT_EQ(round_trip.shape(), t.shape());
        EXPECT_EQ(round_trip.strides(), t.strides());
    }
}

TEST(Expand, AddsDimensionWithStrideZero) {
    auto t = iota_shaped({3});
    auto e = t.expand({2, 3});
    EXPECT_EQ(e.shape(), (Shape{2, 3}));
    EXPECT_EQ(e.strides(), (Strides{0, 1}));
    EXPECT_TRUE(e.shares_storage(t));
    EXPECT_EQ(values<std::int64_t>(e), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));
}

TEST(Expand, SizeOneDimension) {
    auto t = iota_shaped({2, 1});
    auto e = t.expand({2, 4});
    EXPECT_EQ(e.strides(), (Strides{1, 0}));
    EXPECT_EQ(values<std::int64_t>(e), (std::vector<std::int64_t>{0, 0, 0, 0, 1, 1, 1, 1}));
}

TEST(Expand, MinusOneKeepsDimension) {
    auto t = iota_shaped({2, 1});
    EXPECT_EQ(t.expand({-1, 4}).shape(), (Shape{2, 4}));
    EXPECT_EQ(t.expand({3, -1, -1}).shape(), (Shape{3, 2, 1}));
}

TEST(Expand, Errors) {
    auto t = iota_shaped({2, 3});
    EXPECT_THROW((void)t.expand({3, 3}), kael::Error);
    EXPECT_THROW((void)t.expand({3}), kael::Error);
    EXPECT_THROW((void)t.expand({-1, 2, 3}), kael::Error);
}

TEST(Expand, HasInternalOverlap) {
    auto t = iota_shaped({3});
    EXPECT_FALSE(t.has_internal_overlap());
    EXPECT_TRUE(t.expand({2, 3}).has_internal_overlap());
    EXPECT_FALSE(t.expand({1, 3}).has_internal_overlap());
}

TEST(Slice, StepExample) {
    auto t = iota_shaped({10});
    auto s = t.slice(0, 2, 9, 3);
    EXPECT_EQ(s.shape(), (Shape{3}));
    EXPECT_EQ(s.strides(), (Strides{3}));
    EXPECT_EQ(s.offset(), 2);
    EXPECT_TRUE(s.shares_storage(t));
    EXPECT_EQ(values<std::int64_t>(s), (std::vector<std::int64_t>{2, 5, 8}));
}

TEST(Slice, TwoDimensionalExample) {
    auto s = iota_shaped({4, 5}).slice(1, 1, 4);
    EXPECT_EQ(s.shape(), (Shape{4, 3}));
    EXPECT_EQ(s.strides(), (Strides{5, 1}));
    EXPECT_EQ(s.offset(), 1);
    EXPECT_FALSE(s.is_contiguous());
    EXPECT_EQ(s.at<std::int64_t>({1, 0}), 6);
}

TEST(Slice, NegativeBoundsAndDims) {
    auto t = iota_shaped({5});
    EXPECT_EQ(values<std::int64_t>(t.slice(-1, -3, -1)), (std::vector<std::int64_t>{2, 3}));
}

TEST(Slice, BoundsAreClamped) {
    auto t = iota_shaped({5});
    EXPECT_EQ(t.slice(0, 0, 100).numel(), 5);
    EXPECT_EQ(t.slice(0, -100, 2).numel(), 2);
    EXPECT_EQ(t.slice(0, 3, INT64_MAX).numel(), 2);
}

TEST(Slice, EmptyResult) {
    auto t = iota_shaped({5});
    EXPECT_EQ(t.slice(0, 3, 2).numel(), 0);
    EXPECT_EQ(t.slice(0, 5, 5).numel(), 0);
    EXPECT_TRUE(t.slice(0, 5, 5).is_contiguous());
}

TEST(Slice, BadStepThrows) {
    auto t = iota_shaped({5});
    EXPECT_THROW((void)t.slice(0, 0, 5, 0), kael::Error);
    EXPECT_THROW((void)t.slice(0, 0, 5, -1), kael::Error);
}

TEST(Select, RemovesTheDimension) {
    auto t = iota_shaped({2, 3});
    auto row = t.select(0, 1);
    EXPECT_EQ(row.shape(), (Shape{3}));
    EXPECT_EQ(row.offset(), 3);
    EXPECT_EQ(values<std::int64_t>(row), (std::vector<std::int64_t>{3, 4, 5}));
    EXPECT_EQ(values<std::int64_t>(t.select(1, -1)), (std::vector<std::int64_t>{2, 5}));
}

TEST(Select, OutOfRangeThrows) {
    auto t = iota_shaped({2, 3});
    EXPECT_THROW((void)t.select(0, 2), kael::Error);
    EXPECT_THROW((void)t.select(0, -3), kael::Error);
    EXPECT_THROW((void)t.select(2, 0), kael::Error);
}

TEST(Narrow, IsAStrictSlice) {
    auto t = iota_shaped({5});
    EXPECT_EQ(values<std::int64_t>(t.narrow(0, 1, 3)), (std::vector<std::int64_t>{1, 2, 3}));
    EXPECT_EQ(values<std::int64_t>(t.narrow(0, -2, 2)), (std::vector<std::int64_t>{3, 4}));
    EXPECT_EQ(t.narrow(0, 5, 0).numel(), 0);
    EXPECT_THROW((void)t.narrow(0, 3, 3), kael::Error);
    EXPECT_THROW((void)t.narrow(0, 6, 0), kael::Error);
    EXPECT_THROW((void)t.narrow(0, 0, -1), kael::Error);
}

TEST(ViewChain, SliceOverTransposeOverSelect) {
    auto t = iota_shaped({3, 4, 5});
    auto chained = t.select(0, 1).transpose(0, 1).slice(0, 1, 5, 2);
    EXPECT_EQ(chained.shape(), (Shape{2, 4}));
    EXPECT_EQ(values<std::int64_t>(chained),
              (std::vector<std::int64_t>{21, 26, 31, 36, 23, 28, 33, 38}));
}

TEST(Flatten, DimensionRanges) {
    auto t = iota_shaped({2, 3, 4});
    EXPECT_EQ(t.flatten().shape(), (Shape{24}));
    EXPECT_EQ(t.flatten(1).shape(), (Shape{2, 12}));
    EXPECT_EQ(t.flatten(0, 1).shape(), (Shape{6, 4}));
    EXPECT_EQ(t.flatten(-2, -1).shape(), (Shape{2, 12}));
    EXPECT_TRUE(t.flatten(1, 1).is_same(t));
    EXPECT_THROW((void)t.flatten(2, 1), kael::Error);
}

TEST(Flatten, ScalarBecomesOneDimensional) {
    EXPECT_EQ(kael::scalar(1).flatten().shape(), (Shape{1}));
}

TEST(Flatten, NonContiguousCopiesInLogicalOrder) {
    auto tr = iota_shaped({2, 3}).T();
    EXPECT_EQ(values<std::int64_t>(tr.flatten()), (std::vector<std::int64_t>{0, 3, 1, 4, 2, 5}));
}

TEST(Unflatten, InverseOfFlatten) {
    auto t = iota_shaped({2, 12});
    EXPECT_EQ(t.unflatten(1, {3, 4}).shape(), (Shape{2, 3, 4}));
    EXPECT_EQ(t.unflatten(-1, {-1, 4}).shape(), (Shape{2, 3, 4}));
    EXPECT_TRUE(t.unflatten(1, {3, 4}).shares_storage(t));
    EXPECT_THROW((void)t.unflatten(1, {5, 5}), kael::Error);
}

TEST(Split, LastPieceCanBeSmaller) {
    auto t = iota_shaped({5, 2});
    auto pieces = t.split(2);
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_EQ(pieces[0].shape(), (Shape{2, 2}));
    EXPECT_EQ(pieces[2].shape(), (Shape{1, 2}));
    for (const auto& piece : pieces) {
        EXPECT_TRUE(piece.shares_storage(t));
    }
    EXPECT_EQ(values<std::int64_t>(pieces[2]), (std::vector<std::int64_t>{8, 9}));
}

TEST(Split, AlongOtherDimAndErrors) {
    auto t = iota_shaped({2, 5});
    EXPECT_EQ(t.split(3, -1)[1].shape(), (Shape{2, 2}));
    EXPECT_THROW((void)t.split(0), kael::Error);
    EXPECT_EQ(iota_shaped({0}).split(2).size(), 1u);
}

TEST(Chunk, MatchesPyTorchSizes) {
    auto pieces = iota_shaped({10}).chunk(3);
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_EQ(pieces[0].numel(), 4);
    EXPECT_EQ(pieces[1].numel(), 4);
    EXPECT_EQ(pieces[2].numel(), 2);
}

TEST(Chunk, CanReturnFewerChunks) {
    EXPECT_EQ(iota_shaped({2}).chunk(5).size(), 2u);
}

TEST(Chunk, EmptyDimensionGivesEmptyChunks) {
    auto pieces = iota_shaped({0}).chunk(3);
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_EQ(pieces[0].numel(), 0);
    EXPECT_THROW((void)iota_shaped({3}).chunk(0), kael::Error);
}

namespace {

struct ViewOp {
    enum Kind { Transpose, Permute, Slice, Select, Unsqueeze, Squeeze, Expand, Reshape };
    Kind kind;
    std::vector<std::int64_t> args;
};

struct ViewCase {
    Shape initial;
    std::vector<ViewOp> ops;
    Shape expected_shape;
    std::vector<std::int64_t> expected_values;
};

Tensor apply(const Tensor& t, const ViewOp& op) {
    switch (op.kind) {
    case ViewOp::Transpose:
        return t.transpose(op.args[0], op.args[1]);
    case ViewOp::Permute:
        return t.permute(op.args);
    case ViewOp::Slice:
        return t.slice(op.args[0], op.args[1], op.args[2], op.args[3]);
    case ViewOp::Select:
        return t.select(op.args[0], op.args[1]);
    case ViewOp::Unsqueeze:
        return t.unsqueeze(op.args[0]);
    case ViewOp::Squeeze:
        return t.squeeze(op.args[0]);
    case ViewOp::Expand:
        return t.expand(op.args);
    case ViewOp::Reshape:
        return t.reshape(op.args);
    }
    throw kael::Error("unknown view operation");
}

} // namespace

TEST(ViewChain, GeneratedNumPyCases) {
    const std::vector<ViewCase> cases = {
#include "generated/view_cases.inc"
    };

    for (std::size_t i = 0; i < cases.size(); ++i) {
        SCOPED_TRACE("case " + std::to_string(i));
        Tensor t = iota_shaped(cases[i].initial);
        for (const ViewOp& op : cases[i].ops) {
            t = apply(t, op);
        }
        EXPECT_EQ(t.shape(), cases[i].expected_shape);
        EXPECT_EQ(values<std::int64_t>(t), cases[i].expected_values);
        EXPECT_EQ(values<std::int64_t>(t.clone()), cases[i].expected_values);
    }
}

TEST(ZeroDimensional, TransposeAndSqueezeAcceptDimZero) {
    auto s = kael::scalar(1);
    EXPECT_TRUE(s.transpose(0, -1).is_same(s));
    EXPECT_TRUE(s.squeeze(0).is_same(s));
    EXPECT_THROW((void)s.transpose(0, 1), kael::Error);
    EXPECT_THROW((void)s.squeeze(1), kael::Error);
}
