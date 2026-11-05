#pragma once

#include <atomic>
#include <iosfwd>
#include <optional>

#include "kael/tensor.h"

namespace kael::autograd {

void backward(const Tensor& root, const std::optional<Tensor>& gradient = std::nullopt,
              bool retain_graph = false);

namespace detail {

[[nodiscard]] std::atomic<std::ostream*>& trace_output();
[[nodiscard]] std::atomic<bool>& anomaly_detection();

} // namespace detail

} // namespace kael::autograd
