#pragma once

#include <optional>

#include "kael/tensor.h"

namespace kael::autograd {

void backward(const Tensor& root, const std::optional<Tensor>& gradient = std::nullopt,
              bool retain_graph = false);

} // namespace kael::autograd
