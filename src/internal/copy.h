#pragma once

#include "kael/tensor.h"

namespace kael::detail {

void copy_strided(Tensor& dst, const Tensor& src);
void convert_strided(Tensor& dst, const Tensor& src);

} // namespace kael::detail
