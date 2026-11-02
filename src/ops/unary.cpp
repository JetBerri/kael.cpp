#include <cmath>
#include <numbers>
#include <type_traits>

#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "autograd/record.h"
#include "internal/arithmetic.h"

namespace kael {

namespace {

template <typename T> T absolute(T x) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return x;
    } else {
        return std::abs(x);
    }
}

template <typename T> T sign_of(T x) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return x;
    } else {
        return static_cast<T>(x > T{0} ? 1 : (x < T{0} ? -1 : 0));
    }
}

template <typename T> T stable_sigmoid(T x) {
    if (x >= 0) {
        return 1 / (1 + std::exp(-x));
    }
    const T e = std::exp(x);
    return e / (1 + e);
}

template <typename T> T exact_gelu(T x) {
    return T(0.5) * x * (1 + std::erf(x / std::numbers::sqrt2_v<T>));
}

template <typename T> T tanh_gelu(T x) {
    const T scale = std::sqrt(2 / std::numbers::pi_v<T>);
    return T(0.5) * x * (1 + std::tanh(scale * (x + T(0.044715) * x * x * x)));
}

} // namespace

Tensor neg(const Tensor& t) {
    KAEL_CHECK(t.dtype() != DType::Bool, "neg does not support dtype bool");
    Tensor out = unary_kernel(t, [](auto x) { return -x; });
    autograd::record_neg(out, t);
    return out;
}

Tensor abs(const Tensor& t) {
    Tensor out = unary_kernel(t, [](auto x) { return absolute(x); });
    autograd::record_abs(out, t);
    return out;
}

Tensor sign(const Tensor& t) {
    return unary_kernel(t, [](auto x) { return sign_of(x); });
}

Tensor exp(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::exp(x); });
    autograd::record_exp(out, t);
    return out;
}

Tensor log(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::log(x); });
    autograd::record_log(out, t);
    return out;
}

Tensor log1p(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::log1p(x); });
    autograd::record_log1p(out, t);
    return out;
}

Tensor expm1(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::expm1(x); });
    autograd::record_expm1(out, t);
    return out;
}

Tensor sqrt(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::sqrt(x); });
    autograd::record_sqrt(out, t);
    return out;
}

Tensor rsqrt(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return 1 / std::sqrt(x); });
    autograd::record_rsqrt(out, t);
    return out;
}

Tensor sin(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::sin(x); });
    autograd::record_sin(out, t);
    return out;
}

Tensor cos(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::cos(x); });
    autograd::record_cos(out, t);
    return out;
}

Tensor tanh(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return std::tanh(x); });
    autograd::record_tanh(out, t);
    return out;
}

Tensor sigmoid(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, [](auto x) { return stable_sigmoid(x); });
    autograd::record_sigmoid(out, t);
    return out;
}

Tensor relu(const Tensor& t) {
    Tensor out = unary_kernel(t, []<typename T>(T x) -> T { return x < T{0} ? T{0} : x; });
    autograd::record_relu(out, t);
    return out;
}

Tensor gelu(const Tensor& t, bool approximate) {
    Tensor out = approximate ? floating_unary_kernel(t, [](auto x) { return tanh_gelu(x); })
                             : floating_unary_kernel(t, [](auto x) { return exact_gelu(x); });
    autograd::record_gelu(out, t, approximate);
    return out;
}

Tensor clamp(const Tensor& t, Scalar min, Scalar max) {
    const DType dtype = promote_with_scalar(promote_with_scalar(t.dtype(), min), max);
    const Tensor input = t.to(dtype);
    Tensor out = empty_like(input);
    detail::clamp_into(out, input, min, max);
    autograd::record_clamp(out, t, min, max);
    return out;
}

Tensor operator-(const Tensor& t) {
    return neg(t);
}

} // namespace kael
