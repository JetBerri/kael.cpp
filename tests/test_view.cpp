#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/tensor.h"

using kael::DType;
using kael::Shape;
using kael::Strides;
using kael::Tensor;

namespace {

// A tensor with the given metadata over the storage of `base`.
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

// The values of an int64 tensor in logical order, through its metadata.
std::vector<std::int64_t> logical_values(const Tensor& t) {
    std::vector<std::int64_t> out;
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        const Shape index = kael::index_from_linear(i, t.shape());
        out.push_back(t.at<std::int64_t>(index));
    }
    return out;
}

Tensor iota(const Shape& shape) {
    return kael::arange(0, static_cast<double>(kael::numel(shape)), 1, DType::I64);
}

// 0, 1, 2 ... laid out as `shape`, built without any view function.
Tensor iota_shaped(const Shape& shape) {
    return manual_view(iota(shape), shape, kael::contiguous_strides(shape));
}

} // namespace

// ---- is_contiguous ---------------------------------------------------------

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

// ---- contiguous, clone, to ----------------------------------------------------

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
    EXPECT_EQ(logical_values(c), (std::vector<std::int64_t>{1, 3}));
}

TEST(Clone, OfContiguousViewWithOffset) {
    auto t = iota_shaped({5});
    auto tail = manual_view(t, {2}, {1}, 3);
    EXPECT_EQ(logical_values(tail.clone()), (std::vector<std::int64_t>{3, 4}));
}

TEST(Clone, OfStride0View) {
    auto t = iota_shaped({3});
    auto expanded = manual_view(t, {2, 3}, {0, 1});
    EXPECT_EQ(logical_values(expanded.clone()), (std::vector<std::int64_t>{0, 1, 2, 0, 1, 2}));
}

TEST(To, WorksOnNonContiguous) {
    auto t = iota_shaped({2, 3});
    auto tr = manual_view(t, {3, 2}, {1, 3});
    auto f = tr.to(DType::F64);
    EXPECT_EQ(f.dtype(), DType::F64);
    EXPECT_EQ(f.at<double>({0, 1}), 3.0);
    EXPECT_EQ(f.at<double>({2, 1}), 5.0);
}
