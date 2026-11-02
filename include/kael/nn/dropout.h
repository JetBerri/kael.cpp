#pragma once

#include "kael/nn/module.h"

namespace kael::nn {

class Dropout : public UnaryModule {
public:
    explicit Dropout(double p = 0.5);
    Tensor forward(const Tensor& input) override;

private:
    double p_;
};

} // namespace kael::nn
