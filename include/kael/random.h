#pragma once

#include <cstdint>
#include <random>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/tensor.h"

// Seeded random tensors.
//
// - One global std::mt19937_64 generator, seeded with 0 by default, so every
//   program is reproducible even without calling manual_seed.
// - Not thread-safe (TODO phase 9).
// - The engine is identical everywhere, but the standard distributions are
//   not: libstdc++ (Linux) and libc++ (macOS) may turn the same engine output
//   into different numbers. The same seed gives the same values on the same
//   machine, not necessarily across platforms. TODO: own distributions
//   (Box-Muller for the normal) for full cross-platform reproducibility.

namespace kael {

// Resets the global generator.
void manual_seed(std::uint64_t seed);

// The global generator (internal; for shuffling in data loaders, etc.).
std::mt19937_64& global_generator();

// Uniform in [0, 1). Floating point dtypes only.
Tensor rand(const Shape& shape, DType dtype = DType::F32);

// Normal with mean 0 and standard deviation 1. Floating point dtypes only.
Tensor randn(const Shape& shape, DType dtype = DType::F32);

// Uniform integers in [low, high), int64. Throws if low >= high.
Tensor randint(std::int64_t low, std::int64_t high, const Shape& shape);

// A random permutation of 0 .. n - 1, int64.
Tensor randperm(std::int64_t n);

} // namespace kael
