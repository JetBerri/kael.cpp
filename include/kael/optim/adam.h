#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "kael/optim/optimizer.h"

namespace kael::optim {

struct AdamOptions {
    double learning_rate = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
};

enum class WeightDecayMode { Coupled, Decoupled };

class AdamBase : public Optimizer {
public:
    AdamBase(std::vector<Tensor> parameters, const AdamOptions& options, WeightDecayMode mode);

private:
    void update() override;

    AdamOptions options_;
    WeightDecayMode mode_;
    std::vector<std::int64_t> steps_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
};

class Adam final : public AdamBase {
public:
    Adam(std::vector<Tensor> parameters, const AdamOptions& options = {})
        : AdamBase(std::move(parameters), options, WeightDecayMode::Coupled) {}
};

struct AdamWOptions : AdamOptions {
    AdamWOptions() { weight_decay = 0.01; }
};

class AdamW final : public AdamBase {
public:
    AdamW(std::vector<Tensor> parameters, const AdamWOptions& options = {})
        : AdamBase(std::move(parameters), options, WeightDecayMode::Decoupled) {}
};

} // namespace kael::optim
