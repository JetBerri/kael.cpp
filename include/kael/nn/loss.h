#pragma once

#include "kael/tensor.h"

namespace kael::nn {

enum class Reduction { Mean, Sum, None };

[[nodiscard]] Tensor mse_loss(const Tensor& prediction, const Tensor& target,
                              Reduction reduction = Reduction::Mean);
[[nodiscard]] Tensor nll_loss(const Tensor& log_probabilities, const Tensor& targets,
                              Reduction reduction = Reduction::Mean);
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                                   Reduction reduction = Reduction::Mean);
[[nodiscard]] Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& targets,
                                                      Reduction reduction = Reduction::Mean);

} // namespace kael::nn
