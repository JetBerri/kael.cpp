#include <cstdint>
#include <memory>
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
    EXPECT_EQ(values<std::int64_t>(expanded.clone()), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));
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
