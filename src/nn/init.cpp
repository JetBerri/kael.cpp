#include "kael/nn/init.h"

#include <cmath>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/random.h"

namespace kael::nn::init {

namespace {

double leaky_relu_gain(double a) {
    return std::sqrt(2.0 / (1.0 + a * a));
}

} // namespace

Fans fans_of(const Tensor& weight) {
    KAEL_CHECK_VALUE(weight.dim() >= 2, "fans need a tensor with at least 2 dimensions, got shape "
                                            << to_string(weight.shape()));
    const std::int64_t receptive_field = weight.numel() / (weight.size(0) * weight.size(1));
    return {weight.size(1) * receptive_field, weight.size(0) * receptive_field};
}

Tensor& uniform_(Tensor& t, double low, double high) {
    const autograd::NoGradGuard guard;
    return t.copy_(rand(t.shape(), t.dtype()) * (high - low) + low);
}

Tensor& normal_(Tensor& t, double mean, double std) {
    const autograd::NoGradGuard guard;
    return t.copy_(randn(t.shape(), t.dtype()) * std + mean);
}

Tensor& zeros_(Tensor& t) {
    const autograd::NoGradGuard guard;
    return t.zero_();
}

Tensor& ones_(Tensor& t) {
    const autograd::NoGradGuard guard;
    return t.fill_(1);
}

Tensor& constant_(Tensor& t, Scalar value) {
    const autograd::NoGradGuard guard;
    return t.fill_(value);
}

Tensor& xavier_uniform_(Tensor& t, double gain) {
    const Fans fans = fans_of(t);
    const double limit = gain * std::sqrt(6.0 / static_cast<double>(fans.in + fans.out));
    return uniform_(t, -limit, limit);
}

Tensor& kaiming_uniform_(Tensor& t, double a) {
    const double limit = leaky_relu_gain(a) * std::sqrt(3.0 / static_cast<double>(fans_of(t).in));
    return uniform_(t, -limit, limit);
}

Tensor& kaiming_normal_(Tensor& t, double a) {
    return normal_(t, 0.0, leaky_relu_gain(a) / std::sqrt(static_cast<double>(fans_of(t).in)));
}

} // namespace kael::nn::init
