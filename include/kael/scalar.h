#pragma once

#include <concepts>
#include <cstdint>

#include "kael/dtype.h"

namespace kael {

class Scalar {
public:
    constexpr Scalar(bool value) : value_(value ? 1.0 : 0.0), dtype_(DType::Bool) {}

    template <std::integral T>
    constexpr Scalar(T value) : value_(static_cast<double>(value)), dtype_(DType::I64) {}

    template <std::floating_point T>
    constexpr Scalar(T value) : value_(static_cast<double>(value)), dtype_(DType::F32) {}

    [[nodiscard]] constexpr double value() const noexcept { return value_; }
    [[nodiscard]] constexpr DType dtype() const noexcept { return dtype_; }

private:
    double value_;
    DType dtype_;
};

// A scalar never raises the precision of a tensor within the same category:
// F32 + 2.5 stays F32, while I64 + 2.5 moves to the default float, F32.
constexpr DType promote_with_scalar(DType tensor_dtype, Scalar scalar) {
    const bool same_or_lower_category =
        scalar.dtype() == DType::Bool ||
        (scalar.dtype() == DType::I64 && tensor_dtype != DType::Bool) ||
        is_floating_point(tensor_dtype);
    return same_or_lower_category ? tensor_dtype : scalar.dtype();
}

} // namespace kael
