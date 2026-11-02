#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/kael.h"
#include "kael/nn/activation.h"
#include "kael/nn/dropout.h"
#include "kael/nn/embedding.h"
#include "kael/nn/init.h"
#include "kael/nn/layernorm.h"
#include "kael/nn/linear.h"
#include "kael/nn/loss.h"
#include "kael/nn/sequential.h"

#include "support/allclose.h"
#include "support/npy.h"

using kael::DType;
using kael::Tensor;
using kael::autograd::gradcheck;
using kael::nn::Reduction;
using kael::test::allclose;
using Inputs = std::vector<Tensor>;

namespace {

Tensor golden(const std::string& name) {
    return kael::test::read_npy(std::filesystem::path(KAEL_GOLDEN_DIR) / "nn" / (name + ".npy"));
}

const kael::test::AllcloseOptions kGoldenTolerance{.tolerance =
                                                       kael::test::Tolerance{1e-10, 1e-12}};

Tensor differentiable(const kael::Shape& shape) {
    Tensor t = kael::randn(shape, DType::F64);
    t.requires_grad_();
    return t;
}

double min_of(const Tensor& t) {
    return kael::min(t).item();
}

double max_of(const Tensor& t) {
    return kael::max(t).item();
}

} // namespace

TEST(NnInit, FansOfALinearWeight) {
    const auto fans = kael::nn::init::fans_of(kael::zeros({128, 784}));
    EXPECT_EQ(fans.in, 784);
    EXPECT_EQ(fans.out, 128);
}

TEST(NnInit, FansNeedTwoDimensions) {
    EXPECT_THROW((void)kael::nn::init::fans_of(kael::zeros({3})), kael::Error);
}

TEST(NnInit, UniformStaysInRange) {
    kael::manual_seed(1);
    Tensor t = kael::zeros({1000});
    kael::nn::init::uniform_(t, -2.0, 3.0);
    EXPECT_GE(min_of(t), -2.0);
    EXPECT_LE(max_of(t), 3.0);
    EXPECT_NEAR(kael::mean(t).item(), 0.5, 0.2);
}

TEST(NnInit, NormalHasRequestedMoments) {
    kael::manual_seed(2);
    Tensor t = kael::zeros({20000}, DType::F64);
    kael::nn::init::normal_(t, 1.5, 0.5);
    EXPECT_NEAR(kael::mean(t).item(), 1.5, 0.02);
    EXPECT_NEAR(kael::std(t).item(), 0.5, 0.02);
}

TEST(NnInit, ConstantsFill) {
    Tensor t = kael::full({2}, 7);
    EXPECT_EQ(kael::nn::init::zeros_(t).at<float>({0}), 0.0f);
    EXPECT_EQ(kael::nn::init::ones_(t).at<float>({1}), 1.0f);
    EXPECT_EQ(kael::nn::init::constant_(t, 2.5).at<float>({1}), 2.5f);
}

TEST(NnInit, InitWorksOnParameterRequiringGrad) {
    Tensor t = kael::zeros({2, 2});
    t.requires_grad_();
    EXPECT_NO_THROW(kael::nn::init::normal_(t, 0.0, 1.0));
    EXPECT_TRUE(t.is_leaf());
}

TEST(NnInit, XavierAndKaimingLimits) {
    kael::manual_seed(3);
    Tensor weight = kael::zeros({100, 200});
    kael::nn::init::xavier_uniform_(weight);
    const double xavier = std::sqrt(6.0 / 300.0);
    EXPECT_LE(max_of(weight), xavier);
    EXPECT_GT(max_of(weight), 0.9 * xavier);

    kael::nn::init::kaiming_uniform_(weight);
    const double kaiming = std::sqrt(2.0) * std::sqrt(3.0 / 200.0);
    EXPECT_LE(max_of(weight), kaiming);
    EXPECT_GT(max_of(weight), 0.9 * kaiming);

    kael::nn::init::kaiming_normal_(weight);
    EXPECT_NEAR(kael::std(weight).item(), std::sqrt(2.0) / std::sqrt(200.0), 0.005);
}

TEST(NnLinear, DefaultInitBoundsMatchPyTorch) {
    kael::manual_seed(4);
    const kael::nn::Linear layer(784, 128);
    const double limit = 1.0 / 28.0;
    EXPECT_LE(max_of(layer.weight()), limit);
    EXPECT_GE(min_of(layer.weight()), -limit);
    EXPECT_GT(max_of(layer.weight()), 0.9 * limit);
    EXPECT_LE(max_of(*layer.bias()), limit);
}

TEST(NnLinear, OutputShapes) {
    kael::nn::Linear layer(4, 2);
    EXPECT_EQ(layer(kael::ones({4})).shape(), (kael::Shape{2}));
    EXPECT_EQ(layer(kael::ones({3, 4})).shape(), (kael::Shape{3, 2}));
    EXPECT_EQ(layer(kael::ones({5, 3, 4})).shape(), (kael::Shape{5, 3, 2}));
}

TEST(NnLinear, WithoutBiasRegistersOneParameter) {
    const kael::nn::Linear layer(4, 2, false);
    EXPECT_EQ(layer.parameters().size(), 1u);
    EXPECT_FALSE(layer.bias().has_value());
}

TEST(NnLinear, MatchesPyTorch) {
    kael::nn::Linear layer(3, 2, true, DType::F64);
    layer.load_state_dict({{"weight", golden("linear_weight")}, {"bias", golden("linear_bias")}});
    EXPECT_TRUE(
        allclose(layer(golden("linear_x")).detach(), golden("linear_out"), kGoldenTolerance));
}

TEST(NnLinear, Gradcheck) {
    kael::manual_seed(5);
    kael::nn::Linear layer(3, 2, true, DType::F64);
    const auto result = gradcheck([&](const Inputs& in) { return layer(in[0]); },
                                  {differentiable({4, 3}), layer.weight(), *layer.bias()});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(NnActivation, ModulesCallTheirFunctions) {
    const Tensor x = kael::from_vector<double>({-1, 0.5});
    EXPECT_TRUE(allclose(kael::nn::ReLU{}(x), kael::relu(x)));
    EXPECT_TRUE(allclose(kael::nn::GELU{}(x), kael::gelu(x)));
    EXPECT_TRUE(allclose(kael::nn::GELU{true}(x), kael::gelu(x, true)));
    EXPECT_TRUE(allclose(kael::nn::Sigmoid{}(x), kael::sigmoid(x)));
    EXPECT_TRUE(allclose(kael::nn::Tanh{}(x), kael::tanh(x)));
}

TEST(NnDropout, EvalIsIdentity) {
    kael::nn::Dropout dropout(0.5);
    dropout.eval();
    const Tensor x = kael::ones({10});
    EXPECT_TRUE(dropout(x).is_same(x));
}

TEST(NnDropout, ZeroProbabilityIsIdentity) {
    kael::nn::Dropout dropout(0.0);
    const Tensor x = kael::ones({10});
    EXPECT_TRUE(dropout(x).is_same(x));
}

TEST(NnDropout, TrainKeepsProportionAndExpectedValue) {
    kael::manual_seed(6);
    kael::nn::Dropout dropout(0.2);
    const Tensor out = dropout(kael::ones({100000}));
    const double kept = kael::mean((out != 0).to(DType::F32)).item();
    EXPECT_NEAR(kept, 0.8, 0.01);
    EXPECT_NEAR(max_of(out), 1.25, 1e-6);
    EXPECT_NEAR(kael::mean(out).item(), 1.0, 0.01);
}

TEST(NnDropout, InvalidProbabilityThrows) {
    EXPECT_THROW(kael::nn::Dropout(1.0), kael::Error);
    EXPECT_THROW(kael::nn::Dropout(-0.1), kael::Error);
}

TEST(NnLayerNorm, NormalizesLastDimensions) {
    kael::manual_seed(7);
    kael::nn::LayerNorm norm({8}, 1e-5, DType::F64);
    const Tensor out = norm(kael::randn({4, 8}, DType::F64) * 5.0 + 3.0).detach();
    EXPECT_LT(max_of(kael::abs(kael::mean(out, {-1}))), 1e-10);
    EXPECT_LT(max_of(kael::abs(kael::std(out, {-1}, false, 0) - 1.0)), 1e-3);
}

TEST(NnLayerNorm, MatchesPyTorch) {
    kael::nn::LayerNorm norm({3, 4}, 1e-5, DType::F64);
    norm.load_state_dict({{"weight", golden("norm_weight")}, {"bias", golden("norm_bias")}});
    EXPECT_TRUE(allclose(norm(golden("norm_x")).detach(), golden("norm_out"), kGoldenTolerance));
}

TEST(NnLayerNorm, RejectsWrongTrailingShape) {
    kael::nn::LayerNorm norm({3});
    EXPECT_THROW((void)norm(kael::ones({3, 4})), kael::Error);
}

TEST(NnLayerNorm, Gradcheck) {
    kael::manual_seed(10);
    kael::nn::LayerNorm norm({3}, 1e-5, DType::F64);
    const auto parameters = norm.parameters();
    const auto result = gradcheck([&](const Inputs& in) { return norm(in[0]); },
                                  {differentiable({2, 3}), parameters[0], parameters[1]});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(NnLayerNorm, BackwardReachesParameters) {
    kael::manual_seed(8);
    kael::nn::LayerNorm norm({3}, 1e-5, DType::F64);
    kael::sum(norm(kael::randn({2, 3}, DType::F64))).backward();
    for (const Tensor& parameter : norm.parameters()) {
        EXPECT_TRUE(parameter.grad().has_value());
    }
}

TEST(NnEmbedding, MatchesPyTorch) {
    kael::nn::Embedding embedding(5, 3, DType::F64);
    embedding.load_state_dict({{"weight", golden("embedding_weight")}});
    EXPECT_TRUE(allclose(embedding(golden("embedding_indices")).detach(), golden("embedding_out"),
                         kGoldenTolerance));
}

TEST(NnEmbedding, BackwardAddsRepeatedIndices) {
    kael::nn::Embedding embedding(3, 2, DType::F64);
    const Tensor indices = kael::from_vector<std::int64_t>({1, 1, 2});
    kael::sum(embedding(indices)).backward();
    const Tensor gradient = *embedding.weight().grad();
    EXPECT_EQ(gradient.at<double>({0, 0}), 0.0);
    EXPECT_EQ(gradient.at<double>({1, 1}), 2.0);
    EXPECT_EQ(gradient.at<double>({2, 0}), 1.0);
}

TEST(NnEmbedding, Gradcheck) {
    kael::manual_seed(13);
    kael::nn::Embedding embedding(4, 2, DType::F64);
    const Tensor indices = kael::from_vector<std::int64_t>({0, 3, 3}, {3});
    const auto result = gradcheck([&](const Inputs&) { return embedding(indices); },
                                  {embedding.weight()});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(NnEmbedding, OutOfRangeIndexThrows) {
    kael::nn::Embedding embedding(3, 2);
    EXPECT_THROW((void)embedding(kael::from_vector<std::int64_t>({3})), kael::Error);
}

TEST(NnFlatten, KeepsBatchDimension) {
    kael::nn::Flatten flatten;
    EXPECT_EQ(flatten(kael::ones({2, 1, 28, 28})).shape(), (kael::Shape{2, 784}));
}

class NnLossGolden : public testing::TestWithParam<std::pair<Reduction, std::string>> {};

TEST_P(NnLossGolden, MatchesPyTorch) {
    const auto& [reduction, suffix] = GetParam();
    const Tensor mse =
        kael::nn::mse_loss(golden("loss_prediction"), golden("loss_target"), reduction);
    EXPECT_TRUE(allclose(mse, golden("mse_" + suffix), kGoldenTolerance));

    const Tensor nll = kael::nn::nll_loss(kael::log_softmax(golden("loss_logits"), 1),
                                          golden("loss_classes"), reduction);
    EXPECT_TRUE(allclose(nll, golden("nll_" + suffix), kGoldenTolerance));

    const Tensor entropy =
        kael::nn::cross_entropy(golden("loss_logits"), golden("loss_classes"), reduction);
    EXPECT_TRUE(allclose(entropy, golden("cross_entropy_" + suffix), kGoldenTolerance));

    const Tensor binary = kael::nn::binary_cross_entropy_with_logits(
        golden("loss_binary_logits"), golden("loss_binary_targets"), reduction);
    EXPECT_TRUE(allclose(binary, golden("bce_" + suffix), kGoldenTolerance));
}

INSTANTIATE_TEST_SUITE_P(Reductions, NnLossGolden,
                         testing::Values(std::pair{Reduction::Mean, std::string("mean")},
                                         std::pair{Reduction::Sum, std::string("sum")},
                                         std::pair{Reduction::None, std::string("none")}));

TEST(NnLoss, CrossEntropyWorkedExample) {
    const Tensor logits = kael::from_vector<double>({2, 1, 0.1}, {1, 3});
    const Tensor target = kael::from_vector<std::int64_t>({0});
    EXPECT_NEAR(kael::nn::cross_entropy(logits, target).item(), 0.4170, 1e-3);
}

TEST(NnLoss, UniformPredictionGivesLogOfClasses) {
    const Tensor logits = kael::zeros({4, 10}, DType::F64);
    const Tensor targets = kael::from_vector<std::int64_t>({0, 3, 7, 9});
    EXPECT_NEAR(kael::nn::cross_entropy(logits, targets).item(), std::log(10.0), 1e-12);
}

TEST(NnLoss, TargetsOutOfRangeThrowWithClearMessage) {
    const Tensor logits = kael::zeros({1, 10});
    try {
        (void)kael::nn::cross_entropy(logits, kael::from_vector<std::int64_t>({10}));
        FAIL() << "expected an error";
    } catch (const kael::Error& error) {
        EXPECT_NE(std::string(error.what()).find("target 10 is out of bounds for 10 classes"),
                  std::string::npos);
    }
    EXPECT_THROW((void)kael::nn::cross_entropy(logits, kael::from_vector<std::int64_t>({-1})),
                 kael::Error);
}

TEST(NnLoss, TargetsMustMatchBatch) {
    EXPECT_THROW(
        (void)kael::nn::cross_entropy(kael::zeros({2, 3}), kael::from_vector<std::int64_t>({0})),
        kael::Error);
    EXPECT_THROW((void)kael::nn::nll_loss(kael::zeros({3}), kael::from_vector<std::int64_t>({0})),
                 kael::Error);
}

TEST(NnLoss, BinaryCrossEntropyIsStableForLargeLogits) {
    const Tensor logits = kael::from_vector<double>({1000, -1000});
    const Tensor targets = kael::from_vector<double>({1, 0});
    EXPECT_NEAR(kael::nn::binary_cross_entropy_with_logits(logits, targets).item(), 0.0, 1e-12);
}

TEST(NnLoss, Gradcheck) {
    kael::manual_seed(9);
    const Tensor classes = kael::from_vector<std::int64_t>({0, 2});
    const Tensor binary_targets = kael::from_vector<double>({1, 0, 1, 0, 0, 1}, {2, 3});
    const std::vector<kael::autograd::GradcheckFunction> functions = {
        [](const Inputs& in) { return kael::nn::mse_loss(in[0], in[0] * 0.5 + 1.0); },
        [&](const Inputs& in) { return kael::nn::cross_entropy(in[0], classes); },
        [&](const Inputs& in) {
            return kael::nn::nll_loss(kael::log_softmax(in[0], 1), classes, Reduction::Sum);
        },
        [&](const Inputs& in) {
            return kael::nn::binary_cross_entropy_with_logits(in[0], binary_targets);
        },
    };
    for (const auto& function : functions) {
        const auto result = gradcheck(function, {differentiable({2, 3})});
        EXPECT_TRUE(result.passed) << result.report;
    }
}

namespace {

Tensor composed_cross_entropy(const Tensor& logits, const Tensor& targets,
                              kael::nn::Reduction reduction) {
    return kael::nn::nll_loss(kael::log_softmax(logits, 1), targets, reduction);
}

class CrossEntropyFused : public testing::TestWithParam<kael::DType> {};

} // namespace

TEST_P(CrossEntropyFused, ValueIsExactlyTheComposedValue) {
    kael::manual_seed(31);
    const Tensor logits = kael::randn({40, 13}, GetParam());
    const Tensor targets = kael::randint(0, 13, {40});
    for (const auto reduction : {kael::nn::Reduction::None, kael::nn::Reduction::Sum}) {
        EXPECT_TRUE(kael::equal(kael::nn::cross_entropy(logits, targets, reduction),
                                composed_cross_entropy(logits, targets, reduction)));
    }
}

TEST_P(CrossEntropyFused, GradientMatchesTheComposedGradient) {
    kael::manual_seed(32);
    const Tensor values = kael::randn({40, 13}, GetParam());
    const Tensor targets = kael::randint(0, 13, {40});
    Tensor fused = values.clone().requires_grad_();
    Tensor composed = values.clone().requires_grad_();

    kael::nn::cross_entropy(fused, targets).backward();
    composed_cross_entropy(composed, targets, kael::nn::Reduction::Mean).backward();

    const auto tolerance = GetParam() == kael::DType::F32 ? kael::test::Tolerance{1e-5, 1e-5}
                                                          : kael::test::Tolerance{1e-13, 1e-13};
    EXPECT_TRUE(kael::test::allclose(*fused.grad(), *composed.grad(), {.tolerance = tolerance}));
}

INSTANTIATE_TEST_SUITE_P(Floating, CrossEntropyFused,
                         testing::Values(kael::DType::F32, kael::DType::F64));

TEST(CrossEntropyFusedGraph, RecordsItsOwnNode) {
    Tensor logits = kael::zeros({2, 3}).requires_grad_();
    const Tensor targets = kael::from_vector<std::int64_t>({0, 2}, {2});
    const Tensor losses = kael::nn::cross_entropy(logits, targets, kael::nn::Reduction::None);
    ASSERT_NE(losses.grad_fn(), nullptr);
    EXPECT_EQ(losses.grad_fn()->name(), "CrossEntropyBackward");
}

TEST(CrossEntropyFusedGraph, RejectsTargetsOutOfRange) {
    const Tensor logits = kael::zeros({2, 3});
    const Tensor targets = kael::from_vector<std::int64_t>({0, 3}, {2});
    EXPECT_THROW((void)kael::nn::cross_entropy(logits, targets), kael::Error);
}

TEST(CrossEntropyFusedGraph, GradientOfAnEmptyBatchIsEmpty) {
    Tensor logits = kael::zeros({0, 3}).requires_grad_();
    const Tensor targets = kael::zeros({0}, kael::DType::I64);
    const Tensor losses = kael::nn::cross_entropy(logits, targets, kael::nn::Reduction::Sum);
    losses.backward();
    EXPECT_EQ(logits.grad()->numel(), 0);
}

TEST(CrossEntropyFusedGraph, ThePublicOpRejectsInvalidClasses) {
    const Tensor logits = kael::zeros({2, 3});
    EXPECT_THROW((void)kael::cross_entropy_losses(
                     logits, kael::from_vector<std::int64_t>({0, 1000000}, {2})),
                 kael::IndexError);
    EXPECT_THROW((void)kael::cross_entropy_losses(
                     logits, kael::from_vector<std::int64_t>({-1, 0}, {2})),
                 kael::IndexError);
    EXPECT_THROW((void)kael::cross_entropy_losses(logits, kael::zeros({2})), kael::TypeError);
    EXPECT_THROW((void)kael::cross_entropy_losses(
                     kael::zeros({3}), kael::from_vector<std::int64_t>({0}, {1})),
                 kael::ValueError);
}

TEST(CrossEntropyFusedGraph, GradientOfANonContiguousViewMatchesACopy) {
    kael::manual_seed(35);
    const Tensor values = kael::randn({6, 4});
    const Tensor targets = kael::randint(0, 6, {4});
    Tensor leaf = values.clone().requires_grad_();
    kael::nn::cross_entropy(leaf.T(), targets).backward();

    Tensor copied = values.T().contiguous().requires_grad_();
    kael::nn::cross_entropy(copied, targets).backward();

    EXPECT_TRUE(kael::test::allclose(leaf.grad()->T(), *copied.grad()));
}
