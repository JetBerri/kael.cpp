#pragma once

#include "kael/tensor.h"

namespace kael::nn::functional {

[[nodiscard]] Tensor relu(const Tensor& input);
[[nodiscard]] Tensor gelu(const Tensor& input, bool approximate = false);
[[nodiscard]] Tensor sigmoid(const Tensor& input);
[[nodiscard]] Tensor tanh(const Tensor& input);
[[nodiscard]] Tensor softmax(const Tensor& input, std::int64_t dim);
[[nodiscard]] Tensor log_softmax(const Tensor& input, std::int64_t dim);
[[nodiscard]] Tensor dropout(const Tensor& input, double p, bool training);

} // namespace kael::nn::functional
