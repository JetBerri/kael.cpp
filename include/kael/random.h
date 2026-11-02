#pragma once

#include <cstdint>
#include <random>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael {

void manual_seed(std::uint64_t seed);
std::mt19937_64& global_generator();

Tensor rand(const Shape& shape, DType dtype = DType::F32);
Tensor randn(const Shape& shape, DType dtype = DType::F32);
Tensor randint(std::int64_t low, std::int64_t high, const Shape& shape);
Tensor randperm(std::int64_t n);
Tensor bernoulli(const Shape& shape, double probability, DType dtype = DType::F32);
Tensor multinomial(const Tensor& weights, std::int64_t samples, bool replacement = false);

} // namespace kael
