#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>

#include "kael/device_kernels.h"

#if defined(__CUDACC__)
#define KAEL_HD __host__ __device__
#else
#define KAEL_HD
#endif

namespace kael::functions {

using device::ElementOp;
using device::ElementParameters;

template <ElementOp Op> struct Tagged {
    static constexpr ElementOp kOp = Op;
    [[nodiscard]] KAEL_HD ElementParameters parameters() const { return {}; }
};

template <typename T> KAEL_HD T nan_aware_max(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) ? a : b;
        }
    }
    return a < b ? b : a;
}

template <typename T> KAEL_HD T nan_aware_min(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) ? a : b;
        }
    }
    return b < a ? b : a;
}

// Signed overflow is undefined behavior; integer tensors wrap around like PyTorch's.
template <typename T> constexpr bool kWraps = std::is_integral_v<T> && std::is_signed_v<T>;

template <typename T> KAEL_HD T wrapping_add(T a, T b) {
    if constexpr (kWraps<T>) {
        using Wrapping = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<Wrapping>(a) + static_cast<Wrapping>(b));
    } else {
        return a + b;
    }
}

template <typename T> KAEL_HD T wrapping_subtract(T a, T b) {
    if constexpr (kWraps<T>) {
        using Wrapping = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<Wrapping>(a) - static_cast<Wrapping>(b));
    } else {
        return a - b;
    }
}

template <typename T> KAEL_HD T wrapping_multiply(T a, T b) {
    if constexpr (kWraps<T>) {
        using Wrapping = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<Wrapping>(a) * static_cast<Wrapping>(b));
    } else {
        return a * b;
    }
}

template <typename T> KAEL_HD T power_of_negative_exponent(T base, T exponent) {
    if (base == T{1}) {
        return base;
    }
    if (base == static_cast<T>(-1)) {
        return exponent % 2 == 0 ? T{1} : base;
    }
    return T{0};
}

template <typename T> KAEL_HD T integer_power(T base, T exponent) {
    if (exponent < 0) {
        return power_of_negative_exponent(base, exponent);
    }
    using Wrapping = std::make_unsigned_t<T>;
    Wrapping result = 1;
    auto factor = static_cast<Wrapping>(base);
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = static_cast<Wrapping>(result * factor);
        }
        exponent /= 2;
        if (exponent > 0) {
            factor = static_cast<Wrapping>(factor * factor);
        }
    }
    return static_cast<T>(result);
}

template <typename T> KAEL_HD T clamp_value(T x, T low, T high) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(low) || std::isnan(high)) {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }
    const T raised = x < low ? low : x;
    return raised > high ? high : raised;
}

struct Plus : Tagged<ElementOp::Plus> {
    template <typename T> KAEL_HD auto operator()(T a, T b) const {
        if constexpr (kWraps<T>) {
            return wrapping_add(a, b);
        } else {
            return a + b;
        }
    }
};
struct Minus : Tagged<ElementOp::Minus> {
    template <typename T> KAEL_HD auto operator()(T a, T b) const {
        if constexpr (kWraps<T>) {
            return wrapping_subtract(a, b);
        } else {
            return a - b;
        }
    }
};
struct Times : Tagged<ElementOp::Times> {
    template <typename T> KAEL_HD auto operator()(T a, T b) const {
        if constexpr (kWraps<T>) {
            return wrapping_multiply(a, b);
        } else {
            return a * b;
        }
    }
};
struct Divides : Tagged<ElementOp::Divides> {
    template <typename T> KAEL_HD auto operator()(T a, T b) const { return a / b; }
};
struct Power : Tagged<ElementOp::Power> {
    template <typename T> KAEL_HD T operator()(T base, T exponent) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::pow(base, exponent);
        } else {
            return integer_power(base, exponent);
        }
    }
};
struct Maximum : Tagged<ElementOp::Maximum> {
    template <typename T> KAEL_HD T operator()(T a, T b) const { return nan_aware_max(a, b); }
};
struct Minimum : Tagged<ElementOp::Minimum> {
    template <typename T> KAEL_HD T operator()(T a, T b) const { return nan_aware_min(a, b); }
};

struct Equal : Tagged<ElementOp::Equal> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a == b; }
};
struct NotEqual : Tagged<ElementOp::NotEqual> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a != b; }
};
struct Less : Tagged<ElementOp::Less> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a < b; }
};
struct LessEqual : Tagged<ElementOp::LessEqual> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a <= b; }
};
struct Greater : Tagged<ElementOp::Greater> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a > b; }
};
struct GreaterEqual : Tagged<ElementOp::GreaterEqual> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a >= b; }
};
struct LogicalAnd : Tagged<ElementOp::LogicalAnd> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a != T{0} && b != T{0}; }
};
struct LogicalOr : Tagged<ElementOp::LogicalOr> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const { return a != T{0} || b != T{0}; }
};
struct LogicalXor : Tagged<ElementOp::LogicalXor> {
    template <typename T> KAEL_HD bool operator()(T a, T b) const {
        return (a != T{0}) != (b != T{0});
    }
};
struct IsZero : Tagged<ElementOp::IsZero> {
    template <typename T> KAEL_HD bool operator()(T x) const { return x == T{0}; }
};

struct Negate : Tagged<ElementOp::Negate> {
    template <typename T> KAEL_HD auto operator()(T x) const {
        if constexpr (kWraps<T>) {
            return wrapping_subtract(T{0}, x);
        } else {
            return -x;
        }
    }
};
struct Absolute : Tagged<ElementOp::Absolute> {
    template <typename T> KAEL_HD T operator()(T x) const {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            return x;
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::fabs(x);
        } else {
            return x < T{0} ? wrapping_subtract(T{0}, x) : x;
        }
    }
};
struct Sign : Tagged<ElementOp::Sign> {
    template <typename T> KAEL_HD T operator()(T x) const {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            return x;
        } else {
            return static_cast<T>(x > T{0} ? 1 : (x < T{0} ? -1 : 0));
        }
    }
};
struct Exp : Tagged<ElementOp::Exp> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::exp(x); }
};
struct Log : Tagged<ElementOp::Log> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::log(x); }
};
struct Log1p : Tagged<ElementOp::Log1p> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::log1p(x); }
};
struct Expm1 : Tagged<ElementOp::Expm1> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::expm1(x); }
};
struct Sqrt : Tagged<ElementOp::Sqrt> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::sqrt(x); }
};
struct Rsqrt : Tagged<ElementOp::Rsqrt> {
    template <typename T> KAEL_HD T operator()(T x) const { return 1 / std::sqrt(x); }
};
struct Sin : Tagged<ElementOp::Sin> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::sin(x); }
};
struct Cos : Tagged<ElementOp::Cos> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::cos(x); }
};
struct Tanh : Tagged<ElementOp::Tanh> {
    template <typename T> KAEL_HD T operator()(T x) const { return std::tanh(x); }
};
struct Sigmoid : Tagged<ElementOp::Sigmoid> {
    template <typename T> KAEL_HD T operator()(T x) const {
        if (x >= 0) {
            return 1 / (1 + std::exp(-x));
        }
        const T e = std::exp(x);
        return e / (1 + e);
    }
};
struct Relu : Tagged<ElementOp::Relu> {
    template <typename T> KAEL_HD T operator()(T x) const { return x < T{0} ? T{0} : x; }
};
struct GeluExact : Tagged<ElementOp::GeluExact> {
    template <typename T> KAEL_HD T operator()(T x) const {
        return T(0.5) * x * (1 + std::erf(x / std::numbers::sqrt2_v<T>));
    }
};
struct GeluTanh : Tagged<ElementOp::GeluTanh> {
    template <typename T> KAEL_HD T operator()(T x) const {
        const T scale = std::sqrt(2 / std::numbers::pi_v<T>);
        return T(0.5) * x * (1 + std::tanh(scale * (x + T(0.044715) * x * x * x)));
    }
};

template <typename T> struct Clamp {
    static constexpr ElementOp kOp = ElementOp::Clamp;
    T low;
    T high;
    [[nodiscard]] KAEL_HD ElementParameters parameters() const {
        return {static_cast<double>(low), static_cast<double>(high)};
    }
    KAEL_HD T operator()(T x) const { return clamp_value(x, low, high); }
};

struct Where : Tagged<ElementOp::Where> {
    template <typename T> KAEL_HD T operator()(T keep, T a, T b) const {
        return keep != T{0} ? a : b;
    }
};

template <typename T> struct MaskedFill {
    static constexpr ElementOp kOp = ElementOp::MaskedFill;
    T replacement;
    [[nodiscard]] KAEL_HD ElementParameters parameters() const {
        return {static_cast<double>(replacement), 0.0};
    }
    KAEL_HD T operator()(T current, T selected) const {
        return selected != T{0} ? replacement : current;
    }
};

} // namespace kael::functions
