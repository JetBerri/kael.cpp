#pragma once

#include "kael/nn/functional.h"
#include "kael/nn/module.h"

namespace kael::nn {

class ReLU : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return functional::relu(input); }
};

class GELU : public UnaryModule {
public:
    explicit GELU(bool approximate = false) : approximate_(approximate) {}
    Tensor forward(const Tensor& input) override { return functional::gelu(input, approximate_); }

private:
    bool approximate_;
};

class Sigmoid : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return functional::sigmoid(input); }
};

class Tanh : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return functional::tanh(input); }
};

} // namespace kael::nn
