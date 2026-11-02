#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/kernels.h"
#include "kael/operand.h"
#include "kael/tensor.h"

namespace kael::detail {

inline constexpr auto kPlus = [](auto a, auto b) { return a + b; };
inline constexpr auto kMinus = [](auto a, auto b) { return a - b; };
inline constexpr auto kTimes = [](auto a, auto b) { return a * b; };
inline constexpr auto kDivides = [](auto a, auto b) { return a / b; };

template <typename T> T power_of_negative_exponent(T base, T exponent) {
    if (base == T{1}) {
        return base;
    }
    if (base == static_cast<T>(-1)) {
        return exponent % 2 == 0 ? T{1} : base;
    }
    return T{0};
}

template <typename T> T integer_power(T base, T exponent) {
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

inline constexpr auto kPower = []<typename T>(T base, T exponent) -> T {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return integer_power(base, exponent);
    }
};

template <typename T> T clamp_value(T x, T low, T high) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(low) || std::isnan(high)) {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }
    const T raised = x < low ? low : x;
    return raised > high ? high : raised;
}

inline void clamp_into(Tensor& out, const Tensor& input, Scalar min, Scalar max) {
    TensorIterator iterator;
    iterator.add_output(out).add_input(input).build();
    dispatch(out.dtype(), [&]<typename T>() {
        auto clamp = [low = min.as<T>(), high = max.as<T>()](T x) {
            return clamp_value(x, low, high);
        };
        run_typed<T, T, 1>(iterator, clamp);
    });
}

inline void check_no_bool(const Tensor& a, const Tensor& b, const char* op) {
    KAEL_CHECK(a.dtype() != DType::Bool && b.dtype() != DType::Bool,
               op << " does not support dtype bool");
}

inline void check_integer_exponent(const Tensor& base, const Operand& exponent) {
    if (exponent.is_tensor() || is_floating_point(base.dtype())) {
        return;
    }
    KAEL_CHECK(is_floating_point(exponent.scalar().dtype()) ||
                   exponent.scalar().as<std::int64_t>() >= 0,
               "integers to negative integer powers are not allowed");
}

inline DType add_dtype(const Tensor& a, const Tensor& b) {
    return a.dtype() == DType::Bool && b.dtype() == DType::Bool ? DType::I64
                                                                : result_dtype({&a, &b});
}

inline DType div_dtype(const Tensor& a, const Tensor& b) {
    const DType promoted = result_dtype({&a, &b});
    return is_floating_point(promoted) ? promoted : DType::F32;
}

} // namespace kael::detail
