#include <memory>

#include <gtest/gtest.h>

#include "kael/kael.h"
#include "kael/nn/activation.h"
#include "kael/nn/linear.h"
#include "kael/nn/loss.h"
#include "kael/nn/sequential.h"
#include "kael/optim/adam.h"
#include "kael/optim/sgd.h"

using kael::DType;
using kael::Tensor;
using kael::nn::Linear;
using kael::nn::ReLU;
using kael::nn::Sequential;

namespace {

double train_small_classifier(std::uint64_t seed) {
    kael::manual_seed(seed);
    Sequential model{std::make_shared<Linear>(8, 16), std::make_shared<ReLU>(),
                     std::make_shared<Linear>(16, 3)};
    const Tensor inputs = kael::randn({16, 8});
    const Tensor targets = kael::randint(0, 3, {16});
    kael::optim::Adam optimizer(model.parameters());
    double loss_value = 0.0;
    for (int step = 0; step < 20; ++step) {
        const Tensor loss = kael::nn::cross_entropy(model(inputs), targets);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        loss_value = loss.item();
    }
    return loss_value;
}

} // namespace

TEST(Training, OverfitsASingleBatch) {
    kael::manual_seed(21);
    Sequential model{std::make_shared<Linear>(20, 64), std::make_shared<ReLU>(),
                     std::make_shared<Linear>(64, 4)};
    const Tensor inputs = kael::randn({32, 20});
    const Tensor targets = kael::randint(0, 4, {32});
    kael::optim::Adam optimizer(model.parameters(), {.learning_rate = 0.01});

    const double initial = kael::nn::cross_entropy(model(inputs), targets).item();
    EXPECT_NEAR(initial, std::log(4.0), 0.5);

    Tensor loss = kael::zeros({});
    for (int step = 0; step < 200; ++step) {
        loss = kael::nn::cross_entropy(model(inputs), targets);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    EXPECT_LT(loss.item(), 0.02);
}

TEST(Training, LinearRegressionConverges) {
    kael::manual_seed(22);
    Linear model(1, 1);
    const Tensor x = kael::randn({200, 1});
    const Tensor noise = kael::randn({200, 1}) * 0.05;
    const Tensor y = x * 3.0 + 2.0 + noise;
    kael::optim::SGD optimizer(model.parameters(), {.learning_rate = 0.1});
    for (int step = 0; step < 200; ++step) {
        const Tensor loss = kael::nn::mse_loss(model(x), y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    EXPECT_NEAR(model.weight().at<float>({0, 0}), 3.0, 0.05);
    EXPECT_NEAR(model.bias()->at<float>({0}), 2.0, 0.05);
}

TEST(Training, SolvesXor) {
    kael::manual_seed(23);
    Sequential model{std::make_shared<Linear>(2, 8), std::make_shared<kael::nn::Tanh>(),
                     std::make_shared<Linear>(8, 2)};
    const Tensor inputs = kael::from_vector<float>({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    const Tensor targets = kael::from_vector<std::int64_t>({0, 1, 1, 0});
    kael::optim::Adam optimizer(model.parameters(), {.learning_rate = 0.05});
    for (int step = 0; step < 300; ++step) {
        const Tensor loss = kael::nn::cross_entropy(model(inputs), targets);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    const kael::autograd::NoGradGuard guard;
    EXPECT_TRUE(kael::equal(kael::argmax(model(inputs), 1), targets));
}

TEST(Training, SameSeedGivesTheSameLoss) {
    EXPECT_EQ(train_small_classifier(24), train_small_classifier(24));
    EXPECT_NE(train_small_classifier(24), train_small_classifier(25));
}
