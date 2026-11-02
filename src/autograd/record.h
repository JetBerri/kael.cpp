#pragma once

#include <cstdint>
#include <vector>

#include "kael/scalar.h"
#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael::autograd {

void record_add(const Tensor& out, const Tensor& a, const Tensor& b);
void record_sub(const Tensor& out, const Tensor& a, const Tensor& b);
void record_mul(const Tensor& out, const Tensor& a, const Tensor& b);
void record_div(const Tensor& out, const Tensor& a, const Tensor& b);
void record_pow(const Tensor& out, const Tensor& base, const Tensor& exponent);
void record_maximum(const Tensor& out, const Tensor& a, const Tensor& b);
void record_minimum(const Tensor& out, const Tensor& a, const Tensor& b);

void record_neg(const Tensor& out, const Tensor& x);
void record_abs(const Tensor& out, const Tensor& x);
void record_exp(const Tensor& out, const Tensor& x);
void record_log(const Tensor& out, const Tensor& x);
void record_log1p(const Tensor& out, const Tensor& x);
void record_expm1(const Tensor& out, const Tensor& x);
void record_sqrt(const Tensor& out, const Tensor& x);
void record_rsqrt(const Tensor& out, const Tensor& x);
void record_sin(const Tensor& out, const Tensor& x);
void record_cos(const Tensor& out, const Tensor& x);
void record_tanh(const Tensor& out, const Tensor& x);
void record_sigmoid(const Tensor& out, const Tensor& x);
void record_relu(const Tensor& out, const Tensor& x);
void record_gelu(const Tensor& out, const Tensor& x, bool approximate);
void record_clamp(const Tensor& out, const Tensor& x, Scalar min, Scalar max);

void record_sum(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim);
void record_prod(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim);
void record_mean(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim);
void record_extreme(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim,
                    const char* name);
void record_var(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim,
                std::int64_t correction);
void record_norm(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim);
void record_logsumexp(const Tensor& out, const Tensor& x, std::int64_t dim, bool keepdim);
void record_softmax(const Tensor& out, const Tensor& x, std::int64_t dim);
void record_log_softmax(const Tensor& out, const Tensor& x, std::int64_t dim);

void record_view(const Tensor& out, const Tensor& x);
void record_transpose(const Tensor& out, const Tensor& x, std::int64_t dim0, std::int64_t dim1);
void record_permute(const Tensor& out, const Tensor& x, const Dims& dims);
void record_squeeze(const Tensor& out, const Tensor& x);
void record_squeeze(const Tensor& out, const Tensor& x, std::int64_t dim);
void record_unsqueeze(const Tensor& out, const Tensor& x, std::int64_t dim);
void record_expand(const Tensor& out, const Tensor& x);
void record_slice(const Tensor& out, const Tensor& x, std::int64_t dim, std::int64_t start,
                  std::int64_t end, std::int64_t step);
void record_select(const Tensor& out, const Tensor& x, std::int64_t dim, std::int64_t index);
void record_clone(const Tensor& out, const Tensor& x);
void record_to(const Tensor& out, const Tensor& x);

void record_cat(const Tensor& out, const std::vector<Tensor>& inputs, std::int64_t dim);
void record_where(const Tensor& out, const Tensor& condition, const Tensor& a, const Tensor& b);
void record_masked_fill(const Tensor& out, const Tensor& x, const Tensor& mask);
void record_index_select(const Tensor& out, const Tensor& x, std::int64_t dim,
                         const Tensor& indices);
void record_gather(const Tensor& out, const Tensor& x, std::int64_t dim, const Tensor& indices);

void record_matmul(const Tensor& out, const Tensor& a, const Tensor& b);

} // namespace kael::autograd
