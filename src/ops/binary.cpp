#include <algorithm>
#include <cmath>
#include <type_traits>

#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

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
    const DType dtype =
        other.is_tensor() ? promote_with_scalar(other.tensor().dtype(), value) : value.dtype();
    return scalar(value.value(), dtype);
}

template <typename T> T floor_division(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::floor(a / b);
    } else {
        KAEL_CHECK(b != 0, "floor_divide by zero with integer dtype");
        const T quotient = static_cast<T>(a / b);
        const bool rounded_towards_zero = (a % b != 0) && ((a < 0) != (b < 0));
        return rounded_towards_zero ? quotient - 1 : quotient;
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

template <typename F> Tensor elementwise_extreme(const Operand& a, const Operand& b, F f) {
    const auto [x, y] = detail::resolve_operands(a, b);
    return binary_kernel(x, y, f);
}

} // namespace

namespace detail {

std::pair<Tensor, Tensor> resolve_operands(const Operand& a, const Operand& b) {
    return {materialize(a, b), materialize(b, a)};
}

} // namespace detail

Tensor add(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    const DType dtype = detail::add_dtype(x.dtype(), y.dtype());
    return binary_kernel(x.to(dtype), y.to(dtype), detail::kPlus);
}

Tensor sub(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    detail::check_no_bool(x, y, "sub");
    return binary_kernel(x, y, detail::kMinus);
}

Tensor mul(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    return binary_kernel(x, y, detail::kTimes);
}

Tensor div(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    const DType dtype = detail::div_dtype(x.dtype(), y.dtype());
    return binary_kernel(x.to(dtype), y.to(dtype), detail::kDivides);
}

Tensor floor_divide(const Operand& a, const Operand& b) {
    const auto [x, y] = detail::resolve_operands(a, b);
    detail::check_no_bool(x, y, "floor_divide");
    return binary_kernel(x, y, [](auto p, auto q) { return floor_division(p, q); });
}

Tensor pow(const Operand& base, const Operand& exponent) {
    const auto [x, y] = detail::resolve_operands(base, exponent);
    detail::check_no_bool(x, y, "pow");
    return binary_kernel(x, y, detail::kPower);
}

Tensor maximum(const Operand& a, const Operand& b) {
    return elementwise_extreme(a, b, [](auto p, auto q) { return detail::nan_aware_max(p, q); });
}

Tensor minimum(const Operand& a, const Operand& b) {
    return elementwise_extreme(a, b, [](auto p, auto q) { return detail::nan_aware_min(p, q); });
}

Tensor eq(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p == q; });
}

Tensor ne(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p != q; });
}

Tensor lt(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p < q; });
}

Tensor le(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p <= q; });
}

Tensor gt(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p > q; });
}

Tensor ge(const Operand& a, const Operand& b) {
    return compare(a, b, [](auto p, auto q) { return p >= q; });
}

Tensor logical_and(const Operand& a, const Operand& b) {
    return logical(a, b, [](auto p, auto q) { return p != 0 && q != 0; });
}

Tensor logical_or(const Operand& a, const Operand& b) {
    return logical(a, b, [](auto p, auto q) { return p != 0 || q != 0; });
}

Tensor logical_xor(const Operand& a, const Operand& b) {
    return logical(a, b, [](auto p, auto q) { return (p != 0) != (q != 0); });
}

Tensor logical_not(const Tensor& t) {
    return unary_kernel(t.to(DType::Bool), [](auto x) { return x == 0; });
}

bool equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        return false;
    }
    const Tensor same = eq(a, b);
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
