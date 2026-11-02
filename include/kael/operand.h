#pragma once

#include <concepts>
#include <variant>

#include "kael/scalar.h"
#include "kael/tensor.h"

namespace kael {

class Operand {
public:
    Operand(const Tensor& tensor) : value_(tensor) {}

    template <std::convertible_to<Scalar> T> Operand(T scalar) : value_(Scalar(scalar)) {}

    [[nodiscard]] bool is_tensor() const noexcept { return std::holds_alternative<Tensor>(value_); }
    [[nodiscard]] const Tensor& tensor() const { return std::get<Tensor>(value_); }
    [[nodiscard]] Scalar scalar() const { return std::get<Scalar>(value_); }

private:
    std::variant<Tensor, Scalar> value_;
};

} // namespace kael
