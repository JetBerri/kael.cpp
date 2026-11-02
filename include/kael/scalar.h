#pragma once

#include <concepts>
#include <cstdint>
#include <variant>

#include "kael/dtype.h"

namespace kael {

class Scalar {
public:
    constexpr Scalar(bool value) : value_(std::int64_t{value ? 1 : 0}), dtype_(DType::Bool) {}

    template <std::integral T>
    constexpr Scalar(T value) : value_(static_cast<std::int64_t>(value)), dtype_(DType::I64) {}

    template <std::floating_point T>
    constexpr Scalar(T value) : value_(static_cast<double>(value)), dtype_(DType::F32) {}

    [[nodiscard]] constexpr DType dtype() const noexcept { return dtype_; }

    template <typename T> [[nodiscard]] constexpr T as() const {
        return std::visit([](auto value) { return convert_value<T>(value); }, value_);
    }

private:
    std::variant<std::int64_t, double> value_;
    DType dtype_;
};

// A scalar never raises the precision of a tensor within the same category:
// F32 + 2.5 stays F32, while I64 + 2.5 moves to the default float, F32.
constexpr DType promote_with_scalar(DType tensor_dtype, Scalar scalar) {
    return dtype_category(scalar.dtype()) <= dtype_category(tensor_dtype) ? tensor_dtype
                                                                          : scalar.dtype();
}

} // namespace kael
