#pragma once

#include <cstdint>

#include "kael/optim/optimizer.h"

namespace kael::optim {

class LRScheduler {
public:
    LRScheduler(const LRScheduler&) = delete;
    LRScheduler& operator=(const LRScheduler&) = delete;
    virtual ~LRScheduler() = default;

    void step();
    [[nodiscard]] double get_lr() const noexcept { return optimizer_.learning_rate(); }

protected:
    explicit LRScheduler(Optimizer& optimizer);

    void apply_schedule();
    [[nodiscard]] double base_lr() const noexcept { return base_lr_; }

private:
    [[nodiscard]] virtual double lr_at(std::int64_t step) const = 0;

    Optimizer& optimizer_;
    double base_lr_;
    std::int64_t step_count_ = 0;
};

class StepLR final : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, std::int64_t step_size, double gamma);

private:
    [[nodiscard]] double lr_at(std::int64_t step) const override;

    std::int64_t step_size_;
    double gamma_;
};

class CosineAnnealingLR final : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, std::int64_t total_steps, double min_lr = 0.0);

private:
    [[nodiscard]] double lr_at(std::int64_t step) const override;

    std::int64_t total_steps_;
    double min_lr_;
};

class LinearWarmup final : public LRScheduler {
public:
    LinearWarmup(Optimizer& optimizer, std::int64_t warmup_steps);

private:
    [[nodiscard]] double lr_at(std::int64_t step) const override;

    std::int64_t warmup_steps_;
};

class WarmupCosineLR final : public LRScheduler {
public:
    WarmupCosineLR(Optimizer& optimizer, std::int64_t warmup_steps, std::int64_t total_steps,
                   double min_lr = 0.0);

private:
    [[nodiscard]] double lr_at(std::int64_t step) const override;

    std::int64_t warmup_steps_;
    std::int64_t total_steps_;
    double min_lr_;
};

} // namespace kael::optim
