#pragma once

#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael::autograd {

[[nodiscard]] Tensor sum_to_shape(const Tensor& gradient, const Shape& target);

} // namespace kael::autograd
