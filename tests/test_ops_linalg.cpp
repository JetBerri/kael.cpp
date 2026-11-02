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

TEST(Matmul, MatrixMatrix) {
    auto a = kael::from_vector({1.f, 2.f, 3.f, 4.f}, {2, 2});
    auto b = kael::from_vector({5.f, 6.f, 7.f, 8.f}, {2, 2});
    EXPECT_EQ(values<float>(kael::matmul(a, b)), (std::vector<float>{19, 22, 43, 50}));
}

TEST(Matmul, ShapeRules) {
    EXPECT_EQ(kael::matmul(kael::zeros({4}), kael::zeros({4})).shape(), (Shape{}));
    EXPECT_EQ(kael::matmul(kael::zeros({2, 3}), kael::zeros({3, 5})).shape(), (Shape{2, 5}));
    EXPECT_EQ(kael::matmul(kael::zeros({3}), kael::zeros({3, 5})).shape(), (Shape{5}));
    EXPECT_EQ(kael::matmul(kael::zeros({2, 3}), kael::zeros({3})).shape(), (Shape{2}));
    EXPECT_EQ(kael::matmul(kael::zeros({2, 1, 3, 4}), kael::zeros({5, 4, 2})).shape(),
              (Shape{2, 5, 3, 2}));
}

TEST(Matmul, InnerSizeZeroGivesZeros) {
    auto out = kael::matmul(kael::zeros({2, 0}), kael::zeros({0, 3}));
    EXPECT_EQ(out.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(out), std::vector<float>(6, 0.f));
}

TEST(Matmul, Errors) {
    EXPECT_THROW(kael::matmul(kael::zeros({2, 3}), kael::zeros({2, 3})), kael::Error);
    EXPECT_THROW(kael::matmul(kael::scalar(1), kael::zeros({2})), kael::Error);
    EXPECT_THROW(kael::matmul(kael::zeros({2}, DType::Bool), kael::zeros({2}, DType::Bool)),
                 kael::Error);
    EXPECT_THROW(kael::matmul(kael::zeros({2, 2, 2}), kael::zeros({3, 2, 2})), kael::Error);
}

TEST(Matmul, PromotesAndAcceptsViews) {
    auto a = kael::arange(0, 4, 1, DType::I64).reshape({2, 2}).T();
    auto b = kael::ones({2, 2});
    auto out = kael::matmul(a, b);
    EXPECT_EQ(out.dtype(), DType::F32);
    EXPECT_EQ(values<float>(out), (std::vector<float>{2, 2, 4, 4}));
    EXPECT_EQ(values<float>(a.matmul(b)), values<float>(out));
}

TEST(Dot, OnlyVectors) {
    auto v = kael::from_vector({1.f, 2.f, 3.f});
    EXPECT_EQ(kael::dot(v, v).item<float>(), 14.f);
    EXPECT_THROW(kael::dot(kael::zeros({2, 2}), v), kael::Error);
    EXPECT_THROW(kael::dot(v, kael::zeros({2})), kael::Error);
}

TEST(Outer, BuildsMatrix) {
    auto out = kael::outer(kael::from_vector({1.f, 2.f}), kael::from_vector({3.f, 4.f, 5.f}));
    EXPECT_EQ(out.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(out), (std::vector<float>{3, 4, 5, 6, 8, 10}));
    EXPECT_THROW(kael::outer(kael::zeros({2, 2}), kael::zeros({2})), kael::Error);
}

TEST(StrictAliases, RejectOtherRanks) {
    EXPECT_EQ(kael::mm(kael::zeros({2, 3}), kael::zeros({3, 4})).shape(), (Shape{2, 4}));
    EXPECT_THROW(kael::mm(kael::zeros({3}), kael::zeros({3, 4})), kael::Error);
    EXPECT_EQ(kael::bmm(kael::zeros({5, 2, 3}), kael::zeros({5, 3, 4})).shape(), (Shape{5, 2, 4}));
    EXPECT_THROW(kael::bmm(kael::zeros({2, 3}), kael::zeros({3, 4})), kael::Error);
}
