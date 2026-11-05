#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/kael.h"
#include "kael/optim/adam.h"
#include "kael/optim/clip.h"
#include "kael/optim/scheduler.h"
#include "kael/optim/sgd.h"

#include "support/allclose.h"
#include "support/npy.h"

using kael::DType;
using kael::Tensor;
using kael::optim::Adam;
using kael::optim::AdamOptions;
using kael::optim::AdamW;
using kael::optim::AdamWOptions;
using kael::optim::Optimizer;
using kael::optim::SGD;
using kael::optim::SgdOptions;
using kael::test::allclose;

namespace {

Tensor golden(const std::string& name) {
    return kael::test::read_npy(std::filesystem::path(KAEL_GOLDEN_DIR) / "optim" / (name + ".npy"));
}

const kael::test::AllcloseOptions kGoldenTolerance{.tolerance =
                                                       kael::test::Tolerance{1e-10, 1e-12}};

Tensor parameter_from(const Tensor& values) {
    Tensor parameter = values.clone();
    parameter.requires_grad_();
    return parameter;
}

template <typename MakeOptimizer>
void expect_trajectory(const std::string& name, MakeOptimizer make_optimizer) {
    Tensor parameter = parameter_from(golden("start"));
    const Tensor coefficients = golden("coefficients");
    auto optimizer = make_optimizer(std::vector<Tensor>{parameter});
    for (int step = 1; step <= 5; ++step) {
        optimizer->zero_grad();
        const Tensor loss =
            kael::sum(parameter * coefficients) + kael::sum(parameter * parameter) * 0.5;
        loss.backward();
        optimizer->step();
        if (step == 1 || step == 5) {
            EXPECT_TRUE(allclose(parameter.detach(), golden(name + "_" + std::to_string(step)),
                                 kGoldenTolerance))
                << name << " step " << step;
        }
    }
}

std::unique_ptr<SGD> make_sgd(std::vector<Tensor> parameters, const SgdOptions& options) {
    return std::make_unique<SGD>(std::move(parameters), options);
}

void expect_rates(kael::optim::LRScheduler& scheduler, const Tensor& rates) {
    for (std::int64_t i = 0; i < rates.numel(); ++i) {
        EXPECT_NEAR(scheduler.get_lr(), rates.at<double>({i}), 1e-12) << "step " << i;
        scheduler.step();
    }
}

std::vector<Tensor> one_parameter() {
    Tensor parameter = kael::zeros({1}, DType::F64);
    parameter.requires_grad_();
    return {parameter};
}

} // namespace

TEST(OptimGolden, SgdVariantsMatchPyTorch) {
    expect_trajectory("sgd", [](auto p) { return make_sgd(p, {.learning_rate = 0.1}); });
    expect_trajectory("sgd_momentum",
                      [](auto p) { return make_sgd(p, {.learning_rate = 0.1, .momentum = 0.9}); });
    expect_trajectory("sgd_nesterov", [](auto p) {
        return make_sgd(p, {.learning_rate = 0.1, .momentum = 0.9, .nesterov = true});
    });
    expect_trajectory("sgd_decay", [](auto p) {
        return make_sgd(p, {.learning_rate = 0.1, .momentum = 0.5, .weight_decay = 0.01});
    });
}

TEST(OptimGolden, AdamVariantsMatchPyTorch) {
    expect_trajectory("adam", [](auto p) {
        return std::make_unique<Adam>(p, AdamOptions{.learning_rate = 0.01});
    });
    expect_trajectory("adam_decay", [](auto p) {
        return std::make_unique<Adam>(p, AdamOptions{.learning_rate = 0.01, .weight_decay = 0.1});
    });
    expect_trajectory("adamw", [](auto p) {
        AdamWOptions options;
        options.learning_rate = 0.01;
        return std::make_unique<AdamW>(p, options);
    });
}

TEST(Optimizer, AdamFirstStepMovesByLearningRate) {
    Tensor parameter = parameter_from(kael::from_vector<double>({1.0}));
    kael::sum(parameter * 0.5).backward();
    Adam optimizer({parameter}, AdamOptions{.learning_rate = 0.001});
    optimizer.step();
    EXPECT_NEAR(parameter.at<double>({0}), 1.0 - 0.001, 1e-9);
}

TEST(Optimizer, AdamCorrectsTheFirstStepOfALateParameter) {
    Tensor early = parameter_from(kael::from_vector<double>({1.0}));
    Tensor late = parameter_from(kael::from_vector<double>({1.0}));
    Adam optimizer({early, late}, AdamOptions{.learning_rate = 0.001});
    kael::sum(early * 0.5).backward();
    optimizer.step();
    optimizer.zero_grad();
    kael::sum(late * 0.5).backward();
    optimizer.step();
    EXPECT_NEAR(late.at<double>({0}), 1.0 - 0.001, 1e-9);
}

TEST(Optimizer, StepCreatesNoGraphAndKeepsTheSameTensor) {
    Tensor parameter = parameter_from(kael::from_vector<double>({1.0, 2.0}));
    const auto storage = parameter.impl()->storage;
    kael::sum(parameter * parameter).backward();
    SGD optimizer({parameter}, {.learning_rate = 0.1});
    optimizer.step();
    EXPECT_TRUE(parameter.is_leaf());
    EXPECT_TRUE(parameter.requires_grad());
    EXPECT_EQ(parameter.impl()->storage, storage);
    EXPECT_EQ(parameter.at<double>({0}), 0.8);
}

TEST(Optimizer, ParametersWithoutGradAreSkipped) {
    Tensor used = parameter_from(kael::from_vector<double>({1.0}));
    Tensor unused = parameter_from(kael::from_vector<double>({5.0}));
    kael::sum(used).backward();
    SGD optimizer({used, unused}, {.learning_rate = 0.5, .momentum = 0.9});
    Adam adam({used, unused});
    EXPECT_NO_THROW(optimizer.step());
    EXPECT_NO_THROW(adam.step());
    EXPECT_EQ(unused.at<double>({0}), 5.0);
}

TEST(Optimizer, ZeroGradClearsGradients) {
    Tensor parameter = parameter_from(kael::from_vector<double>({1.0}));
    kael::sum(parameter).backward();
    SGD optimizer({parameter}, {.learning_rate = 0.5});
    optimizer.zero_grad();
    EXPECT_FALSE(parameter.grad().has_value());
}

TEST(Optimizer, InvalidConfigurationThrows) {
    EXPECT_THROW(SGD(one_parameter(), {.learning_rate = -1.0}), kael::Error);
    EXPECT_THROW(SGD(one_parameter(), {.learning_rate = 0.1, .nesterov = true}), kael::Error);
    EXPECT_THROW(SGD({kael::zeros({1})}, {.learning_rate = 0.1}), kael::Error);
}

TEST(ClipGradNorm, ScalesWhenAboveTheLimit) {
    Tensor a = parameter_from(kael::from_vector<double>({1.0}));
    Tensor b = parameter_from(kael::from_vector<double>({1.0}));
    kael::sum(a * 3.0 + b * 4.0).backward();
    const double total = kael::optim::clip_grad_norm_({a, b}, 1.0);
    EXPECT_NEAR(total, 5.0, 1e-12);
    EXPECT_NEAR(a.grad()->at<double>({0}), 0.6, 1e-6);
    EXPECT_NEAR(b.grad()->at<double>({0}), 0.8, 1e-6);
}

TEST(ClipGradNorm, LeavesSmallGradientsAlone) {
    Tensor a = parameter_from(kael::from_vector<double>({1.0}));
    kael::sum(a * 0.5).backward();
    EXPECT_NEAR(kael::optim::clip_grad_norm_({a}, 1.0), 0.5, 1e-12);
    EXPECT_EQ(a.grad()->at<double>({0}), 0.5);
}

TEST(Schedulers, StepLrMatchesPyTorch) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.1});
    kael::optim::StepLR scheduler(optimizer, 3, 0.5);
    expect_rates(scheduler, golden("step_lr"));
}

TEST(Schedulers, CosineMatchesPyTorch) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.1});
    kael::optim::CosineAnnealingLR scheduler(optimizer, 10, 0.001);
    expect_rates(scheduler, golden("cosine_lr"));
}

TEST(Schedulers, CosineWorkedExample) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.001});
    kael::optim::CosineAnnealingLR scheduler(optimizer, 10);
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }
    EXPECT_NEAR(optimizer.learning_rate(), 0.0005, 1e-12);
}

TEST(Schedulers, LinearWarmup) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.1});
    kael::optim::LinearWarmup scheduler(optimizer, 4);
    const double expected[] = {0.0, 0.025, 0.05, 0.075, 0.1, 0.1};
    for (const double rate : expected) {
        EXPECT_NEAR(scheduler.get_lr(), rate, 1e-12);
        scheduler.step();
    }
}

TEST(Schedulers, WarmupThenCosine) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.1});
    kael::optim::WarmupCosineLR scheduler(optimizer, 2, 6);
    const double expected[] = {0.0,  0.05,
                               0.1,  0.1 * (1 + std::cos(M_PI / 4)) / 2,
                               0.05, 0.1 * (1 + std::cos(3 * M_PI / 4)) / 2,
                               0.0,  0.0};
    for (const double rate : expected) {
        EXPECT_NEAR(scheduler.get_lr(), rate, 1e-12);
        scheduler.step();
    }
}

TEST(Schedulers, InvalidLengthsThrow) {
    SGD optimizer(one_parameter(), {.learning_rate = 0.1});
    EXPECT_THROW(kael::optim::StepLR(optimizer, 0, 0.5), kael::Error);
    EXPECT_THROW(kael::optim::CosineAnnealingLR(optimizer, 0), kael::Error);
    EXPECT_THROW(kael::optim::LinearWarmup(optimizer, 0), kael::Error);
    EXPECT_THROW(kael::optim::WarmupCosineLR(optimizer, 5, 5), kael::Error);
}
