#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/kael.h"
#include "kael/nn/activation.h"
#include "kael/nn/dropout.h"
#include "kael/nn/linear.h"
#include "kael/nn/module.h"
#include "kael/nn/sequential.h"

#include "support/allclose.h"

using kael::Tensor;
using kael::nn::Linear;

namespace {

class TwoLayers : public kael::nn::Module {
public:
    TwoLayers()
        : first(register_module("first", std::make_shared<Linear>(3, 2))),
          second(register_module("second", std::make_shared<Linear>(2, 1))) {}

    std::shared_ptr<Linear> first;
    std::shared_ptr<Linear> second;
};

} // namespace

TEST(NnModule, ParametersAreCollectedRecursively) {
    const TwoLayers model;
    EXPECT_EQ(model.parameters().size(), 4u);
    EXPECT_EQ(model.num_parameters(), 3 * 2 + 2 + 2 * 1 + 1);
}

TEST(NnModule, NamedParametersUseDottedNamesInOrder) {
    const TwoLayers model;
    const auto named = model.named_parameters();
    ASSERT_EQ(named.size(), 4u);
    EXPECT_EQ(named[0].first, "first.weight");
    EXPECT_EQ(named[1].first, "first.bias");
    EXPECT_EQ(named[2].first, "second.weight");
    EXPECT_EQ(named[3].first, "second.bias");
}

TEST(NnModule, RegisteredParametersRequireGrad) {
    const TwoLayers model;
    for (const Tensor& parameter : model.parameters()) {
        EXPECT_TRUE(parameter.requires_grad());
        EXPECT_TRUE(parameter.is_leaf());
    }
}

TEST(NnModule, TrainingFlagPropagates) {
    TwoLayers model;
    EXPECT_TRUE(model.is_training());
    model.eval();
    EXPECT_FALSE(model.is_training());
    EXPECT_FALSE(model.first->is_training());
    model.train();
    EXPECT_TRUE(model.second->is_training());
}

TEST(NnModule, ZeroGradClearsEveryParameter) {
    TwoLayers model;
    kael::sum((*model.second)((*model.first)(kael::ones({1, 3})))).backward();
    EXPECT_TRUE(model.first->weight().grad().has_value());
    model.zero_grad();
    for (const Tensor& parameter : model.parameters()) {
        EXPECT_FALSE(parameter.grad().has_value());
    }
}

TEST(NnModule, StateDictCopiesWeightsBetweenModels) {
    const TwoLayers source;
    TwoLayers target;
    target.load_state_dict(source.state_dict());
    EXPECT_TRUE(
        kael::test::allclose(target.first->weight().detach(), source.first->weight().detach()));
    EXPECT_TRUE(target.first->weight().requires_grad());
    EXPECT_FALSE(target.first->weight().shares_storage(source.first->weight()));
}

TEST(NnModule, StateDictEntriesAreDetached) {
    const TwoLayers model;
    const auto state = model.state_dict();
    EXPECT_FALSE(state.at("first.weight").requires_grad());
    EXPECT_TRUE(state.at("first.weight").shares_storage(model.first->weight()));
}

TEST(NnModule, LoadStateDictRejectsWrongShape) {
    TwoLayers model;
    auto state = model.state_dict();
    state.insert_or_assign("first.weight", kael::zeros({9, 9}));
    EXPECT_THROW(model.load_state_dict(state), kael::Error);
}

TEST(NnModule, LoadStateDictRejectsMissingName) {
    TwoLayers model;
    auto state = model.state_dict();
    state.erase("second.bias");
    EXPECT_THROW(model.load_state_dict(state), kael::Error);
}

TEST(NnModule, SharedParameterIsListedOnce) {
    class Tied : public kael::nn::Module {
    public:
        Tied() {
            const Tensor shared = kael::zeros({2});
            register_parameter("a", shared);
            register_parameter("b", shared);
        }
    };
    const Tied model;
    EXPECT_EQ(model.parameters().size(), 1u);
    EXPECT_EQ(model.state_dict().size(), 2u);
}

TEST(NnModule, RegisterNullModuleThrows) {
    class Holder : public kael::nn::Module {
    public:
        void add_null() { register_module<kael::nn::Module>("missing", nullptr); }
    };
    Holder holder;
    EXPECT_THROW(holder.add_null(), kael::Error);
}

TEST(NnSequential, NamesLayersByPosition) {
    const kael::nn::Sequential model{std::make_shared<Linear>(3, 2),
                                     std::make_shared<kael::nn::ReLU>(),
                                     std::make_shared<Linear>(2, 1)};
    const auto named = model.named_parameters();
    ASSERT_EQ(named.size(), 4u);
    EXPECT_EQ(named[0].first, "0.weight");
    EXPECT_EQ(named[2].first, "2.weight");
}

TEST(NnSequential, AppliesLayersInOrder) {
    kael::nn::Sequential model{std::make_shared<kael::nn::Flatten>(),
                               std::make_shared<Linear>(6, 4), std::make_shared<kael::nn::Tanh>()};
    EXPECT_EQ(model(kael::ones({5, 2, 3})).shape(), (kael::Shape{5, 4}));
}
