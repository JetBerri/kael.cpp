#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/kael.h"

using kael::Tensor;

TEST(AutogradFields, DefaultsToLeafWithoutGrad) {
    const Tensor t = kael::zeros({2});
    EXPECT_FALSE(t.requires_grad());
    EXPECT_TRUE(t.is_leaf());
    EXPECT_FALSE(t.grad().has_value());
    EXPECT_EQ(t.grad_fn(), nullptr);
}

TEST(AutogradFields, RequiresGradToggles) {
    Tensor t = kael::zeros({2});
    t.requires_grad_();
    EXPECT_TRUE(t.requires_grad());
    t.requires_grad_(false);
    EXPECT_FALSE(t.requires_grad());
}

TEST(AutogradFields, OnlyFloatingPointCanRequireGrad) {
    Tensor integers = kael::zeros({2}, kael::DType::I64);
    Tensor flags = kael::zeros({2}, kael::DType::Bool);
    EXPECT_THROW(integers.requires_grad_(), kael::Error);
    EXPECT_THROW(flags.requires_grad_(), kael::Error);
    EXPECT_NO_THROW(integers.requires_grad_(false));
}

TEST(AutogradFields, ViewDoesNotShareRequiresGradWithoutHistory) {
    Tensor base = kael::zeros({2, 3});
    Tensor view = base.transpose(0, 1);
    base.requires_grad_();
    EXPECT_FALSE(view.requires_grad());
    EXPECT_TRUE(view.shares_storage(base));
}

TEST(AutogradFields, StorageVersionStartsAtZeroAndIsSharedByViews) {
    Tensor base = kael::zeros({2, 3});
    const Tensor view = base.transpose(0, 1);
    EXPECT_EQ(base.version(), 0);
    base.add_(1);
    EXPECT_EQ(base.version(), 1);
    EXPECT_EQ(view.version(), 1);
}

TEST(AutogradFields, InplaceOnLeafRequiringGradThrows) {
    Tensor leaf = kael::zeros({2});
    leaf.requires_grad_();
    EXPECT_THROW(leaf.add_(1), kael::Error);
    EXPECT_THROW(leaf.fill_(1), kael::Error);
    EXPECT_THROW(leaf.zero_(), kael::Error);
    EXPECT_THROW(leaf.clamp_(0, 1), kael::Error);
    EXPECT_EQ(leaf.version(), 0);
}

TEST(AutogradFields, InplaceOnLeafIsAllowedWithoutGrad) {
    Tensor leaf = kael::zeros({2});
    leaf.requires_grad_();
    const kael::autograd::NoGradGuard guard;
    EXPECT_NO_THROW(leaf.add_(1));
    EXPECT_EQ(leaf.version(), 1);
}

TEST(AutogradFields, InplaceWithSourceRequiringGradThrows) {
    Tensor source = kael::ones({2});
    source.requires_grad_();
    Tensor target = kael::zeros({2});
    EXPECT_THROW(target.add_(source), kael::Error);
    EXPECT_THROW(target.copy_(source), kael::Error);
}

TEST(AutogradFields, EveryInplaceOpBumpsVersion) {
    Tensor t = kael::ones({2});
    t.add_(1);
    t.sub_(1);
    t.mul_(1);
    t.div_(1);
    t.pow_(1);
    t.clamp_(0, 2);
    t.copy_(kael::ones({2}));
    t.fill_(1);
    t.zero_();
    t.masked_fill_(kael::ones({2}, kael::DType::Bool), 1);
    t.scatter_add_(0, kael::zeros({2}, kael::DType::I64), kael::ones({2}));
    EXPECT_EQ(t.version(), 11);
}
