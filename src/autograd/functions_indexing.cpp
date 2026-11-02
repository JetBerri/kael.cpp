#include <span>
#include <utility>

#include "kael/autograd/functions.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "record.h"
#include "recording.h"

namespace kael::autograd {

void record_cat(const Tensor& out, const std::vector<Tensor>& inputs, std::int64_t dim) {
    if (!should_record(out, std::span<const Tensor>(inputs))) {
        return;
    }
    std::vector<GradientFn> gradients;
    gradients.reserve(inputs.size());
    std::int64_t start = 0;
    for (const Tensor& input : inputs) {
        const std::int64_t length = input.size(dim);
        gradients.emplace_back([dim, start, length, dtype = input.dtype()](const Tensor& g) {
            return g.narrow(dim, start, length).to(dtype);
        });
        start += length;
    }
    set_history(out, "CatBackward", inputs, std::move(gradients));
}

void record_where(const Tensor& out, const Tensor& condition, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        set_history(out, "WhereBackward", {a, b},
                    {[mask = SavedTensor(condition), shape = a.shape(),
                      dtype = a.dtype()](const Tensor& g) {
                         return sum_to_shape(where(mask.get(), g, 0), shape).to(dtype);
                     },
                     [mask = SavedTensor(condition), shape = b.shape(),
                      dtype = b.dtype()](const Tensor& g) {
                         return sum_to_shape(where(mask.get(), 0, g), shape).to(dtype);
                     }});
    }
}

void record_masked_fill(const Tensor& out, const Tensor& x, const Tensor& mask) {
    if (should_record(out, {&x})) {
        set_history(out, "MaskedFillBackward", {x},
                    {[saved = SavedTensor(mask), shape = x.shape()](const Tensor& g) {
                        return sum_to_shape(masked_fill(g, saved.get(), 0), shape);
                    }});
    }
}

void record_index_select(const Tensor& out, const Tensor& x, std::int64_t dim,
                         const Tensor& indices) {
    if (should_record(out, {&x})) {
        set_history(out, "IndexSelectBackward", {x},
                    {[shape = x.shape(), dtype = x.dtype(), dim,
                      saved = SavedTensor(indices)](const Tensor& g) {
                        Tensor full_gradient = zeros(shape, dtype);
                        full_gradient.index_add_(dim, saved.get(), g);
                        return full_gradient;
                    }});
    }
}

void record_gather(const Tensor& out, const Tensor& x, std::int64_t dim, const Tensor& indices) {
    if (should_record(out, {&x})) {
        set_history(out, "GatherBackward", {x},
                    {[shape = x.shape(), dtype = x.dtype(), dim,
                      saved = SavedTensor(indices)](const Tensor& g) {
                        Tensor full_gradient = zeros(shape, dtype);
                        full_gradient.scatter_add_(dim, saved.get(), g);
                        return full_gradient;
                    }});
    }
}

} // namespace kael::autograd
