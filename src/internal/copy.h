#pragma once

#include "kael/tensor.h"

// Element-wise copies between tensors with arbitrary strides.
// Internal header: not installed, not part of the public API.
namespace kael::detail {

// Copies src into dst in logical order. Both must have the same shape and
// dtype (checked).
void copy_strided(Tensor& dst, const Tensor& src);

// Same, but converting each element to the dtype of dst (see convert_value).
void convert_strided(Tensor& dst, const Tensor& src);

} // namespace kael::detail
