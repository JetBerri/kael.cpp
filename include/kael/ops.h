#pragma once

#include "kael/operand.h"
#include "kael/tensor.h"

namespace kael {

Tensor add(const Operand& a, const Operand& b);
Tensor sub(const Operand& a, const Operand& b);
Tensor mul(const Operand& a, const Operand& b);
Tensor div(const Operand& a, const Operand& b);
Tensor floor_divide(const Operand& a, const Operand& b);
Tensor pow(const Operand& base, const Operand& exponent);
Tensor maximum(const Operand& a, const Operand& b);
Tensor minimum(const Operand& a, const Operand& b);

Tensor eq(const Operand& a, const Operand& b);
Tensor ne(const Operand& a, const Operand& b);
Tensor lt(const Operand& a, const Operand& b);
Tensor le(const Operand& a, const Operand& b);
Tensor gt(const Operand& a, const Operand& b);
Tensor ge(const Operand& a, const Operand& b);

Tensor logical_and(const Operand& a, const Operand& b);
Tensor logical_or(const Operand& a, const Operand& b);
Tensor logical_xor(const Operand& a, const Operand& b);
Tensor logical_not(const Tensor& t);

[[nodiscard]] bool equal(const Tensor& a, const Tensor& b);

Tensor operator+(const Operand& a, const Operand& b);
Tensor operator-(const Operand& a, const Operand& b);
Tensor operator*(const Operand& a, const Operand& b);
Tensor operator/(const Operand& a, const Operand& b);
Tensor operator==(const Operand& a, const Operand& b);
Tensor operator!=(const Operand& a, const Operand& b);
Tensor operator<(const Operand& a, const Operand& b);
Tensor operator<=(const Operand& a, const Operand& b);
Tensor operator>(const Operand& a, const Operand& b);
Tensor operator>=(const Operand& a, const Operand& b);

} // namespace kael
