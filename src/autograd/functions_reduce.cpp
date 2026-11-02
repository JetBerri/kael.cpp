#include <algorithm>
#include <utility>

#include "kael/factory.h"
#include "kael/ops.h"

#include "record.h"
#include "recording.h"

namespace kael::autograd {

namespace {

Dims resolved_dims(const Shape& shape, const Dims& dims) {
    Dims resolved;
    const auto rank = static_cast<std::int64_t>(shape.size());
    if (dims.empty()) {
        for (std::int64_t i = 0; i < rank; ++i) {
            resolved.push_back(i);
        }
    } else if (rank > 0) {
        for (const std::int64_t dim : dims) {
            resolved.push_back(wrap_dim(dim, rank));
        }
    }
    std::sort(resolved.begin(), resolved.end());
    return resolved;
}

Tensor restore_reduced_dims(const Tensor& reduced, const Shape& shape, const Dims& dims,
                            bool keepdim) {
    if (keepdim) {
        return reduced;
    }
    Tensor restored = reduced;
    for (const std::int64_t dim : resolved_dims(shape, dims)) {
        restored = restored.unsqueeze(dim);
    }
    return restored;
}

Tensor expand_reduced(const Tensor& reduced, const Shape& shape, const Dims& dims, bool keepdim) {
    return restore_reduced_dims(reduced, shape, dims, keepdim).expand(shape);
}

std::int64_t reduced_count(const Shape& shape, const Dims& dims) {
    std::int64_t count = 1;
    for (const std::int64_t dim : resolved_dims(shape, dims)) {
        count *= shape[static_cast<std::size_t>(dim)];
    }
    return count;
}

} // namespace

void record_sum(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim) {
    if (should_record(out, {&x})) {
        set_history(out, "SumBackward", {x}, {[shape = x.shape(), dims, keepdim](const Tensor& g) {
                        return expand_reduced(g, shape, dims, keepdim);
                    }});
    }
}

void record_mean(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim) {
    if (should_record(out, {&x})) {
        set_history(out, "MeanBackward", {x}, {[shape = x.shape(), dims, keepdim](const Tensor& g) {
                        return expand_reduced(g, shape, dims, keepdim) /
                               static_cast<double>(reduced_count(shape, dims));
                    }});
    }
}

void record_prod(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim) {
    if (should_record(out, {&x})) {
        set_history(out, "ProdBackward", {x},
                    {[saved = SavedTensor(x), dims, keepdim](const Tensor& g) {
                        const Tensor& values = saved.get();
                        const Tensor is_zero = values == 0;
                        const Tensor zero_count = kael::sum(is_zero, dims, true);
                        const Tensor others = kael::prod(where(is_zero, 1, values), dims, true);
                        const Tensor without_zeros = others / values;
                        const Tensor sole_zero = logical_and(is_zero, zero_count == 1);
                        const Tensor factors =
                            where(zero_count == 0, without_zeros, where(sole_zero, others, 0));
                        return expand_reduced(g, values.shape(), dims, keepdim) * factors;
                    }});
    }
}

void record_extreme(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim,
                    const char* name) {
    if (should_record(out, {&x})) {
        set_history(
            out, name, {x},
            {[saved = SavedTensor(x), y = SavedTensor(out), dims, keepdim](const Tensor& g) {
                const Tensor& values = saved.get();
                const Shape& shape = values.shape();
                const Tensor ties = values == restore_reduced_dims(y.get(), shape, dims, keepdim);
                return expand_reduced(g, shape, dims, keepdim) * ties / kael::sum(ties, dims, true);
            }});
    }
}

void record_var(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim,
                std::int64_t correction) {
    if (should_record(out, {&x})) {
        set_history(out, "VarBackward", {x},
                    {[saved = SavedTensor(x), dims, keepdim, correction](const Tensor& g) {
                        const Tensor& values = saved.get();
                        const std::int64_t count = reduced_count(values.shape(), dims);
                        const auto degrees =
                            static_cast<double>(std::max<std::int64_t>(count - correction, 0));
                        const Tensor deviations = values - kael::mean(values, dims, true);
                        return expand_reduced(g, values.shape(), dims, keepdim) * deviations * 2.0 /
                               degrees;
                    }});
    }
}

void record_norm(const Tensor& out, const Tensor& x, const Dims& dims, bool keepdim) {
    if (should_record(out, {&x})) {
        set_history(
            out, "NormBackward", {x},
            {[saved = SavedTensor(x), y = SavedTensor(out), dims, keepdim](const Tensor& g) {
                const Tensor& values = saved.get();
                const Shape& shape = values.shape();
                const Tensor norm = restore_reduced_dims(y.get(), shape, dims, keepdim);
                return where(norm == 0, 0, expand_reduced(g, shape, dims, keepdim) * values / norm);
            }});
    }
}

void record_logsumexp(const Tensor& out, const Tensor& x, std::int64_t dim, bool keepdim) {
    if (should_record(out, {&x})) {
        set_history(out, "LogsumexpBackward", {x},
                    {[saved = SavedTensor(x), dim, keepdim](const Tensor& g) {
                        const Tensor& values = saved.get();
                        return expand_reduced(g, values.shape(), {dim}, keepdim) *
                               kael::softmax(values, dim);
                    }});
    }
}

void record_softmax(const Tensor& out, const Tensor& x, std::int64_t dim) {
    if (should_record(out, {&x})) {
        set_history(out, "SoftmaxBackward", {x}, {[y = SavedTensor(out), dim](const Tensor& g) {
                        return y.get() * (g - kael::sum(g * y.get(), {dim}, true));
                    }});
    }
}

void record_log_softmax(const Tensor& out, const Tensor& x, std::int64_t dim) {
    if (should_record(out, {&x})) {
        set_history(out, "LogSoftmaxBackward", {x}, {[y = SavedTensor(out), dim](const Tensor& g) {
                        return g - kael::exp(y.get()) * kael::sum(g, {dim}, true);
                    }});
    }
}

void record_standardize(const Tensor& out, const Tensor& x, std::int64_t normalized_dims,
                        double eps) {
    if (should_record(out, {&x})) {
        set_history(
            out, "StandardizeBackward", {x},
            {[saved = SavedTensor(x), y = SavedTensor(out), normalized_dims, eps](const Tensor& g) {
                Dims dims;
                for (std::int64_t i = 1; i <= normalized_dims; ++i) {
                    dims.push_back(-i);
                }
                const Tensor scale = kael::sqrt(kael::var(saved.get(), dims, true, 0) + eps);
                return (g - kael::mean(g, dims, true) -
                        y.get() * kael::mean(g * y.get(), dims, true)) /
                       scale;
            }});
    }
}

} // namespace kael::autograd
