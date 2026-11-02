#pragma once

#include <utility>

#include "kael/scalar.h"
#include "kael/tensor.h"

namespace kael::nn::init {

struct Fans {
    std::int64_t in;
    std::int64_t out;
};

[[nodiscard]] Fans fans_of(const Tensor& weight);

Tensor& uniform_(Tensor& t, double low, double high);
Tensor& normal_(Tensor& t, double mean, double std);
Tensor& zeros_(Tensor& t);
Tensor& ones_(Tensor& t);
Tensor& constant_(Tensor& t, Scalar value);
Tensor& xavier_uniform_(Tensor& t, double gain = 1.0);
Tensor& kaiming_uniform_(Tensor& t, double a = 0.0);
Tensor& kaiming_normal_(Tensor& t, double a = 0.0);

} // namespace kael::nn::init
