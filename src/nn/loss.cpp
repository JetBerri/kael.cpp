#include "kael/nn/loss.h"

#include "kael/nn/functional.h"
#include "kael/ops.h"

namespace kael::nn {

namespace {

Tensor reduce(const Tensor& losses, Reduction reduction) {
    switch (reduction) {
    case Reduction::Mean:
        return kael::mean(losses);
    case Reduction::Sum:
        return kael::sum(losses);
    case Reduction::None:
        return losses;
    }
    detail::unreachable();
}

void check_targets(const Tensor& targets, std::int64_t classes) {
    KAEL_CHECK_TYPE(targets.dtype() == DType::I64,
                    "class targets must have dtype i64, got " << targets.dtype());
    if (targets.numel() == 0) {
        return;
    }
    const auto largest = kael::max(targets).item<std::int64_t>();
    const auto smallest = kael::min(targets).item<std::int64_t>();
    KAEL_CHECK_INDEX(largest < classes,
                     "target " << largest << " is out of bounds for " << classes << " classes");
    KAEL_CHECK_INDEX(smallest >= 0,
                     "target " << smallest << " is out of bounds for " << classes << " classes");
}

} // namespace

Tensor mse_loss(const Tensor& prediction, const Tensor& target, Reduction reduction) {
    const Tensor difference = prediction - target;
    return reduce(difference * difference, reduction);
}

Tensor nll_loss(const Tensor& log_probabilities, const Tensor& targets, Reduction reduction) {
    KAEL_CHECK_VALUE(log_probabilities.dim() == 2,
                     "nll_loss needs log probabilities of shape [N, C], got "
                         << to_string(log_probabilities.shape()));
    KAEL_CHECK_VALUE(targets.dim() == 1 && targets.size(0) == log_probabilities.size(0),
                     "nll_loss needs targets of shape [" << log_probabilities.size(0) << "], got "
                                                         << to_string(targets.shape()));
    check_targets(targets, log_probabilities.size(1));
    const Tensor picked = gather(log_probabilities, 1, targets.unsqueeze(1)).squeeze(1);
    return reduce(-picked, reduction);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction) {
    return nll_loss(functional::log_softmax(logits, 1), targets, reduction);
}

Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& targets,
                                        Reduction reduction) {
    const Tensor losses =
        maximum(logits, 0) - logits * targets + log1p(kael::exp(-kael::abs(logits)));
    return reduce(losses, reduction);
}

} // namespace kael::nn
