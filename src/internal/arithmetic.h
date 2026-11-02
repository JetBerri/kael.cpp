#pragma once

#include <cmath>
#include <type_traits>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/tensor.h"

namespace kael::detail {

inline constexpr auto kPlus = [](auto a, auto b) { return a + b; };
inline constexpr auto kMinus = [](auto a, auto b) { return a - b; };
inline constexpr auto kTimes = [](auto a, auto b) { return a * b; };
inline constexpr auto kDivides = [](auto a, auto b) { return a / b; };

template <typename T>
T integer_power(T base, T exponent) {
    KAEL_CHECK(exponent >= 0, "integers to negative integer powers are not allowed");
    T result = 1;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }
    return result;
}

inline constexpr auto kPower = []<typename T>(T base, T exponent) -> T {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return integer_power(base, exponent);
    }
};

struct Clamper {
    double low;
    double high;

    template <typename T>
    T operator()(T x) const {
        const T raised = x < convert_value<T>(low) ? convert_value<T>(low) : x;
        return raised > convert_value<T>(high) ? convert_value<T>(high) : raised;
    }
};

inline void check_no_bool(const Tensor& a, const Tensor& b, const char* op) {
    KAEL_CHECK(a.dtype() != DType::Bool && b.dtype() != DType::Bool,
               op << " does not support dtype bool");
}

inline DType add_dtype(DType a, DType b) {
    return a == DType::Bool && b == DType::Bool ? DType::I64 : promote_types(a, b);
}

inline DType div_dtype(DType a, DType b) {
    const DType promoted = promote_types(a, b);
    return is_floating_point(promoted) ? promoted : DType::F32;
}

} // namespace kael::detail
