#pragma once

#include <vector>

#include "kael/tensor.h"

namespace kael::optim {

class Optimizer {
public:
    Optimizer(std::vector<Tensor> parameters, double learning_rate);
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    virtual ~Optimizer() = default;

    void step();
    void zero_grad();

    [[nodiscard]] double learning_rate() const noexcept { return learning_rate_; }
    void set_learning_rate(double learning_rate);

protected:
    [[nodiscard]] std::vector<Tensor>& parameters() noexcept { return parameters_; }

private:
    virtual void update() = 0;

    std::vector<Tensor> parameters_;
    double learning_rate_;
};

} // namespace kael::optim
