#pragma once

#include <iosfwd>
#include <string>

#include "kael/tensor.h"

namespace kael {

[[nodiscard]] std::string explain(const Tensor& t);
[[nodiscard]] std::string autograd_graph_dot(const Tensor& t);

void set_backward_trace(std::ostream* output);
void set_anomaly_detection(bool enabled);

} // namespace kael
