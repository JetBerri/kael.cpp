#include <limits>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "autograd/record.h"

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
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return log(sum(x, {dim}, keepdim));
        }
        const Tensor largest = finite_or_zero(max(x, {dim}, true));
        const Tensor shifted_total = log(sum(exp(x - largest), {dim}, true)) + largest;
        return keepdim || x.dim() == 0 ? shifted_total : shifted_total.squeeze(dim);
    });
    autograd::record_logsumexp(out, t, dim, keepdim);
    return out;
}

Tensor softmax(const Tensor& t, std::int64_t dim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return x.clone();
        }
        const Tensor exponentials = exp(shifted_by_max(x, dim));
        return exponentials / sum(exponentials, {dim}, true);
    });
    autograd::record_softmax(out, t, dim);
    return out;
}

Tensor log_softmax(const Tensor& t, std::int64_t dim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return x.clone();
        }
        const Tensor shifted = shifted_by_max(x, dim);
        return shifted - log(sum(exp(shifted), {dim}, true));
    });
    autograd::record_log_softmax(out, t, dim);
    return out;
}

} // namespace kael
