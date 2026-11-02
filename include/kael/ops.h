#pragma once

#include <vector>

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

Tensor neg(const Tensor& t);
Tensor abs(const Tensor& t);
Tensor sign(const Tensor& t);
Tensor exp(const Tensor& t);
Tensor log(const Tensor& t);
Tensor log1p(const Tensor& t);
Tensor expm1(const Tensor& t);
Tensor sqrt(const Tensor& t);
Tensor rsqrt(const Tensor& t);
Tensor sin(const Tensor& t);
Tensor cos(const Tensor& t);
Tensor tanh(const Tensor& t);
Tensor sigmoid(const Tensor& t);
Tensor relu(const Tensor& t);
Tensor gelu(const Tensor& t, bool approximate = false);
Tensor clamp(const Tensor& t, Scalar min, Scalar max);

Tensor sum(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor prod(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor mean(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor max(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor min(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor argmax(const Tensor& t);
Tensor argmax(const Tensor& t, std::int64_t dim, bool keepdim = false);
Tensor argmin(const Tensor& t);
Tensor argmin(const Tensor& t, std::int64_t dim, bool keepdim = false);
Tensor var(const Tensor& t, const Dims& dims = {}, bool keepdim = false,
           std::int64_t correction = 1);
Tensor std(const Tensor& t, const Dims& dims = {}, bool keepdim = false,
           std::int64_t correction = 1);
Tensor all(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor any(const Tensor& t, const Dims& dims = {}, bool keepdim = false);
Tensor norm(const Tensor& t, const Dims& dims = {}, bool keepdim = false);

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor outer(const Tensor& a, const Tensor& b);
Tensor mm(const Tensor& a, const Tensor& b);
Tensor bmm(const Tensor& a, const Tensor& b);

Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dim = 0);
Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dim = 0);
Tensor where(const Tensor& condition, const Operand& a, const Operand& b);
Tensor masked_fill(const Tensor& t, const Tensor& mask, Scalar value);
Tensor index_select(const Tensor& t, std::int64_t dim, const Tensor& indices);
Tensor gather(const Tensor& t, std::int64_t dim, const Tensor& indices);

Tensor logsumexp(const Tensor& t, std::int64_t dim, bool keepdim = false);
Tensor softmax(const Tensor& t, std::int64_t dim);
Tensor log_softmax(const Tensor& t, std::int64_t dim);

Tensor operator-(const Tensor& t);
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
Tensor& operator+=(Tensor& t, const Operand& other);
Tensor& operator-=(Tensor& t, const Operand& other);
Tensor& operator*=(Tensor& t, const Operand& other);
Tensor& operator/=(Tensor& t, const Operand& other);

} // namespace kael
