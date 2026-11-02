#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "kael/kael.h"

#include "support/allclose.h"
#include "support/npy.h"

using kael::Tensor;

namespace {

const std::filesystem::path kMlpDir = std::filesystem::path(KAEL_GOLDEN_DIR) / "mlp";

Tensor load(const std::string& name) {
    return kael::test::read_npy(kMlpDir / (name + ".npy"));
}

Tensor load_parameter(const std::string& name) {
    Tensor parameter = load(name);
    parameter.requires_grad_();
    return parameter;
}

void expect_gradient(const Tensor& parameter, const std::string& name) {
    ASSERT_TRUE(parameter.grad().has_value()) << name;
    EXPECT_TRUE(kael::test::allclose(*parameter.grad(), load(name + "_grad"),
                                     {.tolerance = kael::test::Tolerance{1e-6, 1e-12}}))
        << name;
}

} // namespace

TEST(AutogradGolden, TwoLayerPerceptronMatchesPyTorch) {
    const Tensor w1 = load_parameter("w1");
    const Tensor b1 = load_parameter("b1");
    const Tensor w2 = load_parameter("w2");
    const Tensor b2 = load_parameter("b2");

    const Tensor hidden = kael::tanh(kael::matmul(load("inputs"), w1.T()) + b1);
    const Tensor logits = kael::matmul(hidden, w2.T()) + b2;
    const Tensor log_probabilities = kael::log_softmax(logits, 1);
    const Tensor picked = kael::gather(log_probabilities, 1, load("targets").unsqueeze(1));
    const Tensor loss = -kael::mean(picked);
    loss.backward();

    EXPECT_NEAR(loss.item(), load("loss").item(), 1e-12);
    expect_gradient(w1, "w1");
    expect_gradient(b1, "b1");
    expect_gradient(w2, "w2");
    expect_gradient(b2, "b2");
}
