#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::test::values;

TEST(InPlace, ChangesTheTensorAndKeepsTheStorage) {
    auto t = kael::arange(0, 3);
    const auto storage = t.impl()->storage;
    kael::Tensor& result = t.add_(1);
    EXPECT_EQ(&result, &t);
    EXPECT_EQ(t.impl()->storage, storage);
    EXPECT_EQ(values<float>(t), (std::vector<float>{1, 2, 3}));
}

TEST(InPlace, Chains) {
    auto t = kael::full({2}, 3.0);
    t.mul_(2).add_(1);
    EXPECT_EQ(values<float>(t), (std::vector<float>{7, 7}));
}

TEST(InPlace, AllArithmetic) {
    auto t = kael::full({1}, 8.0);
    EXPECT_EQ(t.sub_(2).item<float>(), 6.f);
    EXPECT_EQ(t.div_(3).item<float>(), 2.f);
    EXPECT_EQ(t.pow_(3).item<float>(), 8.f);
    EXPECT_EQ(t.clamp_(0, 5).item<float>(), 5.f);
}

TEST(InPlace, ViewsSeeTheChange) {
    auto t = kael::zeros({2, 2});
    auto row = t.select(0, 1);
    row.add_(5);
    EXPECT_EQ(values<float>(t), (std::vector<float>{0, 0, 5, 5}));
}

TEST(InPlace, OtherBroadcastsButSelfDoesNot) {
    auto t = kael::zeros({2, 3});
    EXPECT_NO_THROW(t.add_(kael::ones({3})));
    auto small = kael::zeros({3});
    EXPECT_THROW(small.add_(kael::ones({2, 3})), kael::Error);
}

TEST(InPlace, DtypeNeverChanges) {
    auto t = kael::zeros({1});
    t.add_(kael::ones({1}, DType::F64));
    EXPECT_EQ(t.dtype(), DType::F32);
    t.add_(kael::ones({1}, DType::I64));
    EXPECT_EQ(t.dtype(), DType::F32);
}

TEST(InPlace, LosingCategoryThrows) {
    auto ints = kael::zeros({1}, DType::I64);
    EXPECT_THROW(ints.add_(kael::ones({1})), kael::Error);
    EXPECT_THROW(ints.add_(0.5), kael::Error);
    EXPECT_THROW(ints.div_(2), kael::Error);
    EXPECT_THROW(ints.clamp_(-0.5, 0.5), kael::Error);
    EXPECT_NO_THROW(ints.add_(1));

    auto bools = kael::zeros({1}, DType::Bool);
    EXPECT_THROW(bools.add_(bools), kael::Error);
    EXPECT_THROW(bools.sub_(1), kael::Error);
}

TEST(InPlace, ExpandedTensorThrows) {
    auto expanded = kael::zeros({3}).expand({2, 3});
    EXPECT_THROW(expanded.add_(1), kael::Error);
    EXPECT_THROW(expanded.fill_(1), kael::Error);
    EXPECT_THROW(expanded.copy_(kael::ones({2, 3})), kael::Error);
    EXPECT_THROW(expanded.clamp_(0, 1), kael::Error);
    EXPECT_NO_THROW(expanded.clone().add_(1));
}

TEST(InPlace, SelfAsOperand) {
    auto t = kael::arange(1, 4);
    t.mul_(t);
    EXPECT_EQ(values<float>(t), (std::vector<float>{1, 4, 9}));
}

TEST(InPlace, PartiallyOverlappingOperandIsReadBeforeWriting) {
    auto t = kael::arange(0, 5);
    auto tail = t.slice(0, 1, 5);
    tail.add_(t.slice(0, 0, 4));
    EXPECT_EQ(values<float>(t), (std::vector<float>{0, 1, 3, 5, 7}));
}

TEST(Copy, BetweenDifferentStrides) {
    auto destination = kael::zeros({2, 3}).T();
    destination.copy_(kael::arange(0, 6).reshape({3, 2}));
    EXPECT_EQ(values<float>(destination), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(Copy, ConvertsAndBroadcasts) {
    auto destination = kael::zeros({2, 2}, DType::I64);
    destination.copy_(kael::from_vector({1.7f, 2.2f}));
    EXPECT_EQ(values<std::int64_t>(destination), (std::vector<std::int64_t>{1, 2, 1, 2}));
}

TEST(Copy, OverlappingWithItself) {
    auto t = kael::arange(0, 4);
    t.slice(0, 1, 4).copy_(t.slice(0, 0, 3));
    EXPECT_EQ(values<float>(t), (std::vector<float>{0, 0, 1, 2}));
}

TEST(Fill, SetsEveryElement) {
    auto t = kael::zeros({2, 2}).T();
    t.fill_(2.5);
    EXPECT_EQ(values<float>(t), std::vector<float>(4, 2.5f));
    t.zero_();
    EXPECT_EQ(values<float>(t), std::vector<float>(4, 0.f));
}

TEST(Fill, ConvertsToTheDtype) {
    auto bools = kael::zeros({1}, DType::Bool);
    bools.fill_(2.5);
    EXPECT_EQ(bools.item<std::uint8_t>(), 1);
}

TEST(CompoundOperators, CallTheInPlaceVersion) {
    auto t = kael::full({1}, 6.0);
    t += 2;
    EXPECT_EQ(t.item<float>(), 8.f);
    t -= 4;
    EXPECT_EQ(t.item<float>(), 4.f);
    t *= kael::full({1}, 3.0);
    EXPECT_EQ(t.item<float>(), 12.f);
    t /= 2;
    EXPECT_EQ(t.item<float>(), 6.f);
}
