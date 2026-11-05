#include <string>

#include <gtest/gtest.h>

#include "kael/autograd/engine.h"
#include "kael/autograd/node.h"
#include "kael/error.h"
#include "kael/kael.h"

#include "support/allclose.h"
#include "support/values.h"

using kael::Tensor;
using kael::test::allclose;

namespace {

Tensor leaf(std::vector<double> values) {
    Tensor t = kael::from_vector(values);
    t.requires_grad_();
    return t;
}

Tensor doubles(std::vector<double> values) {
    return kael::from_vector(values);
}

} // namespace

TEST(AutogradEngine, ResultOfOpHasHistory) {
    const Tensor x = leaf({1, 2});
    const Tensor y = x + x;
    EXPECT_TRUE(y.requires_grad());
    EXPECT_FALSE(y.is_leaf());
    ASSERT_NE(y.grad_fn(), nullptr);
    EXPECT_EQ(y.grad_fn()->name(), "AddBackward");
}

TEST(AutogradEngine, OpWithoutGradInputsHasNoHistory) {
    const Tensor y = doubles({1, 2}) + doubles({3, 4});
    EXPECT_FALSE(y.requires_grad());
    EXPECT_EQ(y.grad_fn(), nullptr);
}

TEST(AutogradEngine, NoGradGuardRecordsNothing) {
    const Tensor x = leaf({1, 2});
    const kael::autograd::NoGradGuard guard;
    const Tensor y = x * x;
    EXPECT_FALSE(y.requires_grad());
    EXPECT_EQ(y.grad_fn(), nullptr);
}

TEST(AutogradEngine, RepeatedInputAccumulates) {
    const Tensor x = leaf({3, -2});
    (x * x).sum().backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({6, -4})));
}

TEST(AutogradEngine, TripleAddAccumulates) {
    const Tensor x = leaf({3, -2});
    (x + x + x).sum().backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({3, 3})));
}

TEST(AutogradEngine, DiamondWaitsForBothPaths) {
    const Tensor x = leaf({1, 2});
    const Tensor a = x * 2;
    const Tensor b = x * 3;
    (a + b).sum().backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({5, 5})));
}

TEST(AutogradEngine, LeafAsRootReceivesSeed) {
    const Tensor x = leaf({4});
    x.backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({1})));
}

TEST(AutogradEngine, NonScalarNeedsExplicitGradient) {
    const Tensor x = leaf({1, 2});
    const Tensor y = x * 2;
    EXPECT_THROW(y.backward(), kael::Error);
    y.backward(doubles({1, 10}));
    EXPECT_TRUE(allclose(*x.grad(), doubles({2, 20})));
}

TEST(AutogradEngine, ExplicitGradientMustMatchShape) {
    const Tensor x = leaf({1, 2});
    const Tensor y = x * 2;
    EXPECT_THROW(y.backward(doubles({1, 2, 3})), kael::Error);
}

TEST(AutogradEngine, RootWithoutGradThrows) {
    EXPECT_THROW(doubles({1}).backward(), kael::Error);
}

TEST(AutogradEngine, SecondBackwardWithoutRetainThrows) {
    const Tensor x = leaf({1, 2});
    const Tensor y = (x * x).sum();
    y.backward();
    try {
        y.backward();
        FAIL() << "expected an error";
    } catch (const kael::Error& error) {
        EXPECT_NE(std::string(error.what()).find("second time"), std::string::npos);
    }
}

TEST(AutogradEngine, RetainGraphAccumulatesInLeaves) {
    const Tensor x = leaf({1, 2});
    const Tensor y = (x * x).sum();
    y.backward(true);
    y.backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({4, 8})));
}

TEST(AutogradEngine, LeafSharedByTwoGraphsAccumulates) {
    const Tensor x = leaf({1, 2});
    const Tensor first = (x * 2).sum();
    const Tensor second = (x * 3).sum();
    first.backward();
    second.backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({5, 5})));
}

TEST(AutogradEngine, ZeroGradClearsGradient) {
    Tensor x = leaf({1, 2});
    (x * 2).sum().backward();
    x.zero_grad();
    EXPECT_FALSE(x.grad().has_value());
}

TEST(AutogradEngine, GradIsACopyOfThePropagatedGradient) {
    const Tensor x = leaf({1, 2});
    const Tensor a = x + 0;
    (a + x).sum().backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({2, 2})));
    EXPECT_FALSE(x.grad()->requires_grad());
}

TEST(AutogradEngine, InvalidGradientShapeNamesTheNode) {
    class BadNode final : public kael::autograd::Node {
    public:
        using Node::Node;
        [[nodiscard]] std::string name() const override { return "BadBackward"; }
        kael::autograd::GradientList apply(const Tensor&) override {
            return {kael::zeros({2, 3}, kael::DType::F64)};
        }
    };
    const Tensor x = leaf({1, 2, 3});
    const Tensor out = kael::zeros({1}, kael::DType::F64);
    out.impl()->autograd = std::make_unique<kael::autograd::AutogradMeta>();
    out.impl()->autograd->requires_grad = true;
    out.impl()->autograd->grad_fn = std::make_shared<BadNode>(
        std::vector<kael::autograd::NodeInput>{kael::autograd::make_input(x)});
    try {
        out.backward();
        FAIL() << "expected an error";
    } catch (const kael::Error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("BadBackward"), std::string::npos) << message;
        EXPECT_NE(message.find("[2, 3]"), std::string::npos) << message;
        EXPECT_NE(message.find("[3]"), std::string::npos) << message;
    }
}

TEST(AutogradEngine, DetachedSavedOutputDoesNotLeak) {
    std::weak_ptr<kael::autograd::Node> watcher;
    {
        const Tensor x = leaf({1, 2});
        const Tensor y = kael::exp(x);
        watcher = y.grad_fn();
        EXPECT_FALSE(watcher.expired());
    }
    EXPECT_TRUE(watcher.expired());
}

TEST(AutogradEngine, DeepChainBackwardAndDestruction) {
    Tensor x = leaf({1});
    Tensor y = x;
    for (int i = 0; i < 10000; ++i) {
        y = y + 1;
    }
    y.backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({1})));
}

TEST(AutogradEngine, DeepChainDestructionWithoutBackward) {
    const Tensor x = leaf({1});
    {
        Tensor y = x;
        for (int i = 0; i < 50000; ++i) {
            y = y + 1;
        }
    }
    SUCCEED();
}

TEST(AutogradEngine, RetainGradStoresIntermediateGradient) {
    const Tensor x = leaf({1, 2});
    Tensor middle = x * 3;
    middle.retain_grad();
    (middle * 2).sum().backward();
    ASSERT_TRUE(middle.grad().has_value());
    EXPECT_TRUE(allclose(*middle.grad(), doubles({2, 2})));
    EXPECT_TRUE(allclose(*x.grad(), doubles({6, 6})));
}

TEST(AutogradEngine, RetainGradOnTensorWithoutGradThrows) {
    Tensor t = doubles({1});
    EXPECT_THROW(t.retain_grad(), kael::Error);
}

TEST(AutogradDetach, SharesStorageWithoutHistory) {
    const Tensor x = leaf({1, 2});
    const Tensor y = x * 2;
    const Tensor detached = y.detach();
    EXPECT_FALSE(detached.requires_grad());
    EXPECT_EQ(detached.grad_fn(), nullptr);
    EXPECT_TRUE(detached.shares_storage(y));
}

TEST(AutogradDetach, CutsTheGradient) {
    const Tensor x = leaf({1, 2});
    (x.detach() * x).sum().backward();
    EXPECT_TRUE(allclose(*x.grad(), doubles({1, 2})));
}

TEST(AutogradDetach, InPlaceTurnsTensorIntoLeaf) {
    const Tensor x = leaf({1, 2});
    Tensor y = x * 2;
    y.detach_();
    EXPECT_TRUE(y.is_leaf());
    EXPECT_FALSE(y.requires_grad());
}

TEST(AutogradVersion, ModifyingSavedInputThrows) {
    const Tensor x = leaf({1, 2});
    Tensor constant = doubles({3, 4});
    const Tensor y = (x * constant).sum();
    constant.add_(1);
    try {
        y.backward();
        FAIL() << "expected an error";
    } catch (const kael::Error& error) {
        EXPECT_NE(std::string(error.what()).find("inplace"), std::string::npos);
    }
}

TEST(AutogradVersion, InplaceOnIntermediateThrows) {
    const Tensor x = leaf({1, 2});
    Tensor y = kael::exp(x);
    EXPECT_THROW(
        {
            y.add_(1);
            y.sum().backward();
        },
        kael::Error);
}

TEST(AutogradVersion, ModifyingSavedOutputThroughDetachThrows) {
    const Tensor x = leaf({1, 2});
    const Tensor y = kael::exp(x);
    Tensor alias = y.detach();
    alias.add_(1);
    EXPECT_THROW(y.sum().backward(), kael::Error);
}

TEST(AutogradDetach, InPlaceOnALeafInTheGraphSkipsItsGradient) {
    Tensor x = leaf({1, 2});
    const Tensor y = (x * 2).sum();
    x.detach_();
    y.backward();
    EXPECT_FALSE(x.grad().has_value());
}

TEST(AutogradDetach, InPlaceOnARetainedOutputSkipsItsGradient) {
    const Tensor x = leaf({1, 2});
    Tensor y = x * 2;
    y.retain_grad();
    const Tensor z = (y * 3).sum();
    y.detach_();
    z.backward();
    EXPECT_FALSE(y.grad().has_value());
    EXPECT_TRUE(allclose(*x.grad(), doubles({6, 6})));
}
