#include "kael/nn/functional.h"

#include "kael/ops.h"
#include "kael/random.h"

namespace kael::nn::functional {

Tensor relu(const Tensor& input) {
    return kael::relu(input);
}

Tensor gelu(const Tensor& input, bool approximate) {
    return kael::gelu(input, approximate);
}

Tensor sigmoid(const Tensor& input) {
    return kael::sigmoid(input);
}

Tensor tanh(const Tensor& input) {
    return kael::tanh(input);
}

Tensor softmax(const Tensor& input, std::int64_t dim) {
    return kael::softmax(input, dim);
}

Tensor log_softmax(const Tensor& input, std::int64_t dim) {
    return kael::log_softmax(input, dim);
}

Tensor dropout(const Tensor& input, double p, bool training) {
    KAEL_CHECK_VALUE(p >= 0.0 && p < 1.0, "dropout probability must be in [0, 1), got " << p);
    if (!training || p == 0.0) {
        return input;
    }
    const double keep = 1.0 - p;
    return input * bernoulli(input.shape(), keep, input.dtype()) / keep;
}

} // namespace kael::nn::functional
