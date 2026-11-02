#include <algorithm>
#include <cmath>
#include <type_traits>

#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "autograd/record.h"
#include "internal/arithmetic.h"
#include "internal/nan_aware.h"
#include "internal/operands.h"

namespace kael {

namespace {

Tensor materialize(const Operand& operand, const Operand& other) {
    if (operand.is_tensor()) {
        return operand.tensor();
    }
    const Scalar value = operand.scalar();
    if (!other.is_tensor()) {
        return scalar(value, value.dtype());
    }
    const Tensor& tensor = other.tensor();
    const DType dtype = promote_with_scalar(tensor.dtype(), value);
    return tensor.device() == Device::CPU ? scalar(value, dtype)
                                          : full({}, value, dtype, tensor.device());
}

template <typename T> T floating_floor_division(T a, T b) {
    if (b == T{0}) {
        return a / b;
    }
    const T remainder = std::fmod(a, b);
    T quotient = (a - remainder) / b;
    if (remainder != T{0} && (b < T{0}) != (remainder < T{0})) {
        quotient -= T{1};
    }
    if (quotient == T{0}) {
        return std::copysign(T{0}, a / b);
    }
    const T floored = std::floor(quotient);
    return quotient - floored > T(0.5) ? floored + T{1} : floored;
}

template <typename T> T integer_floor_division(T a, T b) {
    KAEL_CHECK(b != T{0}, "floor_divide by zero with integer dtype");
    if (b == static_cast<T>(-1)) {
        using Wrapping = std::make_unsigned_t<T>;
        return static_cast<T>(Wrapping{0} - static_cast<Wrapping>(a));
    }
    const T quotient = static_cast<T>(a / b);
    const bool rounded_towards_zero = (a % b != 0) && ((a < 0) != (b < 0));
    return rounded_towards_zero ? static_cast<T>(quotient - 1) : quotient;
}

template <typename T> T floor_division(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return floating_floor_division(a, b);
    } else {
        return integer_floor_division(a, b);
    }
}

template <typename F> Tensor compare(const Operand& a, const Operand& b, F f) {
    const auto [x, y] = detail::resolve_operands(a, b);
    return comparison_kernel(x, y, f);
}

template <typename F> Tensor logical(const Operand& a, const Operand& b, F f) {
    const auto [x, y] = detail::resolve_operands(a, b);
    return comparison_kernel(x.to(DType::Bool), y.to(DType::Bool), f);
}

} // namespace

namespace detail {

std::pair<Tensor, Tensor> resolve_operands(const Operand& a, const Operand& b) {
    return {materialize(a, b), materialize(b, a)};
}

} // namespace detail

Tensor add(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    const DType dtype = detail::add_dtype(x, y);
    Tensor out = autograd::without_grad(
        [&] { return binary_kernel(x.to(dtype), y.to(dtype), detail::kPlus); });
    autograd::record_add(out, x, y);
    return out;
}

Tensor sub(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    detail::check_no_bool(x, y, "sub");
    Tensor out = binary_kernel(x, y, detail::kMinus);
    autograd::record_sub(out, x, y);
    return out;
}

Tensor mul(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    Tensor out = binary_kernel(x, y, detail::kTimes);
    autograd::record_mul(out, x, y);
    return out;
}

Tensor div(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    const DType dtype = detail::div_dtype(x, y);
    Tensor out = autograd::without_grad(
        [&] { return binary_kernel(x.to(dtype), y.to(dtype), detail::kDivides); });
    autograd::record_div(out, x, y);
    return out;
}

Tensor floor_divide(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    detail::check_no_bool(x, y, "floor_divide");
    return binary_kernel(x, y, [](auto p, auto q) { return floor_division(p, q); });
}

Tensor pow(const Operand& base, const Operand& exponent) {
    const auto [x, y] = detail::resolve_operands(base, exponent);
    detail::check_no_bool(x, y, "pow");
    detail::check_integer_exponent(x, exponent);
    Tensor out = binary_kernel(x, y, detail::kPower);
    autograd::record_pow(out, x, y);
    return out;
}

Tensor maximum(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    Tensor out = binary_kernel(x, y, functions::Maximum{});
    autograd::record_maximum(out, x, y);
    return out;
}

Tensor minimum(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    Tensor out = binary_kernel(x, y, functions::Minimum{});
    autograd::record_minimum(out, x, y);
    return out;
}

Tensor eq(const Operand& a, const Operand& b) {
    return compare(a, b, functions::Equal{});
}

Tensor ne(const Operand& a, const Operand& b) {
    return compare(a, b, functions::NotEqual{});
}

Tensor lt(const Operand& a, const Operand& b) {
    return compare(a, b, functions::Less{});
}

Tensor le(const Operand& a, const Operand& b) {
    return compare(a, b, functions::LessEqual{});
}

Tensor gt(const Operand& a, const Operand& b) {
    return compare(a, b, functions::Greater{});
}

Tensor ge(const Operand& a, const Operand& b) {
    return compare(a, b, functions::GreaterEqual{});
}

Tensor logical_and(const Operand& a, const Operand& b) {
    return logical(a, b, functions::LogicalAnd{});
}

Tensor logical_or(const Operand& a, const Operand& b) {
    return logical(a, b, functions::LogicalOr{});
}

Tensor logical_xor(const Operand& a, const Operand& b) {
    return logical(a, b, functions::LogicalXor{});
}

Tensor logical_not(const Tensor& t) {
    return unary_kernel(t.to(DType::Bool), functions::IsZero{});
}

bool equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        return false;
    }
    const Tensor same = eq(a, b).cpu();
    const std::uint8_t* flags = same.data_ptr<std::uint8_t>();
    return std::all_of(flags, flags + same.numel(), [](std::uint8_t flag) { return flag != 0; });
}

Tensor operator+(const Operand& a, const Operand& b) {
    return add(a, b);
}

Tensor operator-(const Operand& a, const Operand& b) {
    return sub(a, b);
}

Tensor operator*(const Operand& a, const Operand& b) {
    return mul(a, b);
}

Tensor operator/(const Operand& a, const Operand& b) {
    return div(a, b);
}

Tensor operator==(const Operand& a, const Operand& b) {
    return eq(a, b);
}

Tensor operator!=(const Operand& a, const Operand& b) {
    return ne(a, b);
}

Tensor operator<(const Operand& a, const Operand& b) {
    return lt(a, b);
}

Tensor operator<=(const Operand& a, const Operand& b) {
    return le(a, b);
}

Tensor operator>(const Operand& a, const Operand& b) {
    return gt(a, b);
}

Tensor operator>=(const Operand& a, const Operand& b) {
    return ge(a, b);
}

} // namespace kael
