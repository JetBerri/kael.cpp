#pragma once

#include <vector>

#include "kael/tensor.h"

namespace kael::optim {

double clip_grad_norm_(const std::vector<Tensor>& parameters, double max_norm);

} // namespace kael::optim
