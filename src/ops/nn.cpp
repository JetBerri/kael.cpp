#include <limits>

#include "kael/factory.h"
#include "kael/ops.h"

namespace kael {

namespace {

Tensor as_floating(const Tensor& t) {
    return is_floating_point(t.dtype()) ? t : t.to(DType::F32);
}

Tensor finite_or_zero(const Tensor& t) {
    const double infinity = std::numeric_limits<double>::infinity();
    return where(abs(t) == infinity, 0, t);
}

Tensor shifted_by_max(const Tensor& x, std::int64_t dim) {
    return x - max(x, {dim}, true);
}

} // namespace

Tensor logsumexp(const Tensor& t, std::int64_t dim, bool keepdim) {
    const Tensor x = as_floating(t);
    if (x.numel() == 0) {
        return log(sum(x, {dim}, keepdim));
    }
    const Tensor largest = finite_or_zero(max(x, {dim}, true));
    const Tensor shifted_total = log(sum(exp(x - largest), {dim}, true)) + largest;
    return keepdim || x.dim() == 0 ? shifted_total : shifted_total.squeeze(dim);
}

Tensor softmax(const Tensor& t, std::int64_t dim) {
    const Tensor x = as_floating(t);
    if (x.numel() == 0) {
        return x.clone();
    }
    const Tensor exponentials = exp(shifted_by_max(x, dim));
    return exponentials / sum(exponentials, {dim}, true);
}

Tensor log_softmax(const Tensor& t, std::int64_t dim) {
    const Tensor x = as_floating(t);
    if (x.numel() == 0) {
        return x.clone();
    }
    const Tensor shifted = shifted_by_max(x, dim);
    return shifted - log(sum(exp(shifted), {dim}, true));
}

} // namespace kael
