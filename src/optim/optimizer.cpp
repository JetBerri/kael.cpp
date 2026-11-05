#include "kael/optim/optimizer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/optim/adam.h"
#include "kael/optim/clip.h"
#include "kael/optim/scheduler.h"
#include "kael/optim/sgd.h"

namespace kael::optim {

namespace {

double cosine_between(double high, double low, double progress) {
    return low + 0.5 * (high - low) * (1.0 + std::cos(std::numbers::pi * progress));
}

} // namespace

Optimizer::Optimizer(std::vector<Tensor> parameters, double learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    KAEL_CHECK_VALUE(learning_rate >= 0.0,
                     "learning rate must be non-negative, got " << learning_rate);
    for (const Tensor& parameter : parameters_) {
        KAEL_CHECK(parameter.requires_grad() && parameter.is_leaf(),
                   "optimizers need leaf tensors that require grad");
    }
}

void Optimizer::step() {
    const autograd::NoGradGuard guard;
    update();
}

void Optimizer::zero_grad() {
    for (Tensor& parameter : parameters_) {
        parameter.zero_grad();
    }
}

void Optimizer::set_learning_rate(double learning_rate) {
    KAEL_CHECK_VALUE(learning_rate >= 0.0,
                     "learning rate must be non-negative, got " << learning_rate);
    learning_rate_ = learning_rate;
}

SGD::SGD(std::vector<Tensor> parameters, const SgdOptions& options)
    : Optimizer(std::move(parameters), options.learning_rate), options_(options),
      velocities_(this->parameters().size()) {
    KAEL_CHECK_VALUE(!options.nesterov || options.momentum > 0.0,
                     "nesterov needs a positive momentum");
}

void SGD::update() {
    for (std::size_t i = 0; i < parameters().size(); ++i) {
        Tensor& parameter = parameters()[i];
        const std::optional<Tensor> stored = parameter.grad();
        if (!stored) {
            continue;
        }
        Tensor gradient =
            options_.weight_decay > 0.0 ? *stored + parameter * options_.weight_decay : *stored;
        if (options_.momentum > 0.0) {
            std::optional<Tensor>& velocity = velocities_[i];
            if (velocity) {
                velocity->mul_(options_.momentum).add_(gradient);
            } else {
                velocity = gradient.clone();
            }
            gradient = options_.nesterov ? gradient + *velocity * options_.momentum : *velocity;
        }
        parameter.sub_(gradient * learning_rate());
    }
}

AdamBase::AdamBase(std::vector<Tensor> parameters, const AdamOptions& options, WeightDecayMode mode)
    : Optimizer(std::move(parameters), options.learning_rate), options_(options), mode_(mode) {
    for (const Tensor& parameter : this->parameters()) {
        first_moments_.push_back(zeros_like(parameter));
        second_moments_.push_back(zeros_like(parameter));
        steps_.push_back(0);
    }
}

// Steps are counted per parameter, as in PyTorch: a parameter that gets its first gradient late
// still needs the full bias correction of a first step.
void AdamBase::update() {
    for (std::size_t i = 0; i < parameters().size(); ++i) {
        Tensor& parameter = parameters()[i];
        const std::optional<Tensor> stored = parameter.grad();
        if (!stored) {
            continue;
        }
        const auto step = static_cast<double>(++steps_[i]);
        const double first_correction = 1.0 - std::pow(options_.beta1, step);
        const double second_correction = 1.0 - std::pow(options_.beta2, step);
        Tensor gradient = *stored;
        if (options_.weight_decay > 0.0) {
            if (mode_ == WeightDecayMode::Coupled) {
                gradient = gradient + parameter * options_.weight_decay;
            } else {
                parameter.mul_(1.0 - learning_rate() * options_.weight_decay);
            }
        }
        Tensor& first = first_moments_[i];
        Tensor& second = second_moments_[i];
        first.mul_(options_.beta1).add_(gradient * (1.0 - options_.beta1));
        second.mul_(options_.beta2).add_(gradient * gradient * (1.0 - options_.beta2));
        const Tensor corrected_first = first / first_correction;
        const Tensor corrected_second = second / second_correction;
        parameter.sub_(corrected_first / (kael::sqrt(corrected_second) + options_.epsilon) *
                       learning_rate());
    }
}

double clip_grad_norm_(const std::vector<Tensor>& parameters, double max_norm) {
    const autograd::NoGradGuard guard;
    double squared_total = 0.0;
    for (const Tensor& parameter : parameters) {
        if (const auto gradient = parameter.grad()) {
            const double norm = kael::norm(*gradient).item();
            squared_total += norm * norm;
        }
    }
    const double total = std::sqrt(squared_total);
    KAEL_CHECK_VALUE(std::isfinite(total),
                     "the total gradient norm is " << total << ", so it cannot be clipped");
    const double coefficient = max_norm / (total + 1e-6);
    if (coefficient < 1.0) {
        for (const Tensor& parameter : parameters) {
            if (auto gradient = parameter.grad()) {
                gradient->mul_(coefficient);
            }
        }
    }
    return total;
}

LRScheduler::LRScheduler(Optimizer& optimizer)
    : optimizer_(optimizer), base_lr_(optimizer.learning_rate()) {}

void LRScheduler::step() {
    ++step_count_;
    apply_schedule();
}

void LRScheduler::apply_schedule() {
    optimizer_.set_learning_rate(lr_at(step_count_));
}

StepLR::StepLR(Optimizer& optimizer, std::int64_t step_size, double gamma)
    : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma) {
    KAEL_CHECK_VALUE(step_size > 0, "StepLR step size must be positive, got " << step_size);
    apply_schedule();
}

double StepLR::lr_at(std::int64_t step) const {
    const std::int64_t decays = step / step_size_;
    return base_lr() * std::pow(gamma_, static_cast<double>(decays));
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, std::int64_t total_steps, double min_lr)
    : LRScheduler(optimizer), total_steps_(total_steps), min_lr_(min_lr) {
    KAEL_CHECK_VALUE(total_steps > 0,
                     "cosine schedule length must be positive, got " << total_steps);
    apply_schedule();
}

double CosineAnnealingLR::lr_at(std::int64_t step) const {
    const double progress =
        static_cast<double>(std::min(step, total_steps_)) / static_cast<double>(total_steps_);
    return cosine_between(base_lr(), min_lr_, progress);
}

LinearWarmup::LinearWarmup(Optimizer& optimizer, std::int64_t warmup_steps)
    : LRScheduler(optimizer), warmup_steps_(warmup_steps) {
    KAEL_CHECK_VALUE(warmup_steps > 0, "warmup length must be positive, got " << warmup_steps);
    apply_schedule();
}

double LinearWarmup::lr_at(std::int64_t step) const {
    return base_lr() *
           std::min(1.0, static_cast<double>(step + 1) / static_cast<double>(warmup_steps_));
}

WarmupCosineLR::WarmupCosineLR(Optimizer& optimizer, std::int64_t warmup_steps,
                               std::int64_t total_steps, double min_lr)
    : LRScheduler(optimizer), warmup_steps_(warmup_steps), total_steps_(total_steps),
      min_lr_(min_lr) {
    KAEL_CHECK_VALUE(warmup_steps >= 0 && warmup_steps < total_steps,
                     "warmup of " << warmup_steps << " steps does not fit a schedule of "
                                  << total_steps);
    apply_schedule();
}

double WarmupCosineLR::lr_at(std::int64_t step) const {
    if (step < warmup_steps_) {
        return base_lr() * static_cast<double>(step + 1) / static_cast<double>(warmup_steps_);
    }
    const double progress = static_cast<double>(std::min(step, total_steps_) - warmup_steps_) /
                            static_cast<double>(total_steps_ - warmup_steps_);
    return cosine_between(base_lr(), min_lr_, progress);
}

} // namespace kael::optim
