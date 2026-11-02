#include <cmath>
#include <numbers>
#include <type_traits>

#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

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

} // namespace

Tensor neg(const Tensor& t) {
    KAEL_CHECK(t.dtype() != DType::Bool, "neg does not support dtype bool");
    return unary_kernel(t, [](auto x) { return -x; });
}

Tensor abs(const Tensor& t) {
    return unary_kernel(t, [](auto x) { return absolute(x); });
}

Tensor sign(const Tensor& t) {
    return unary_kernel(t, [](auto x) { return sign_of(x); });
}

Tensor exp(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::exp(x); });
}

Tensor log(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::log(x); });
}

Tensor log1p(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::log1p(x); });
}

Tensor expm1(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::expm1(x); });
}

Tensor sqrt(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::sqrt(x); });
}

Tensor rsqrt(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return 1 / std::sqrt(x); });
}

Tensor sin(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::sin(x); });
}

Tensor cos(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::cos(x); });
}

Tensor tanh(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return std::tanh(x); });
}

Tensor sigmoid(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return stable_sigmoid(x); });
}

Tensor relu(const Tensor& t) {
    return unary_kernel(t, []<typename T>(T x) -> T { return x < T{0} ? T{0} : x; });
}

Tensor gelu(const Tensor& t) {
    return floating_unary_kernel(t, [](auto x) { return exact_gelu(x); });
}

Tensor clamp(const Tensor& t, Scalar min, Scalar max) {
    const DType dtype = promote_with_scalar(promote_with_scalar(t.dtype(), min), max);
    const Tensor input = t.to(dtype);
    Tensor out = empty_like(input);
    detail::clamp_into(out, input, min, max);
    return out;
}

Tensor operator-(const Tensor& t) {
    return neg(t);
}

} // namespace kael
