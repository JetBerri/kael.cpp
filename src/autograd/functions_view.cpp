#include "kael/autograd/functions.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "record.h"
#include "recording.h"

namespace kael::autograd {

void record_view(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        set_history(out, "ViewBackward", {x},
                    {[shape = x.shape()](const Tensor& g) { return g.reshape(shape); }});
    }
}

void record_transpose(const Tensor& out, const Tensor& x, std::int64_t dim0, std::int64_t dim1) {
    if (should_record(out, {&x})) {
        set_history(out, "TransposeBackward", {x},
                    {[dim0, dim1](const Tensor& g) { return g.transpose(dim0, dim1); }});
    }
}

void record_permute(const Tensor& out, const Tensor& x, const Dims& dims) {
    if (should_record(out, {&x})) {
        Dims inverse(dims.size());
        for (std::size_t i = 0; i < dims.size(); ++i) {
            inverse[dim_position(dims[i], x.dim())] = static_cast<std::int64_t>(i);
        }
        set_history(out, "PermuteBackward", {x}, {[inverse = std::move(inverse)](const Tensor& g) {
                        return g.permute(inverse);
                    }});
    }
}

void record_squeeze(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        set_history(out, "SqueezeBackward", {x},
                    {[shape = x.shape()](const Tensor& g) { return g.reshape(shape); }});
    }
}

void record_squeeze(const Tensor& out, const Tensor& x, std::int64_t dim) {
    if (should_record(out, {&x})) {
        set_history(out, "SqueezeBackward", {x},
                    {[dim](const Tensor& g) { return g.unsqueeze(dim); }});
    }
}

void record_unsqueeze(const Tensor& out, const Tensor& x, std::int64_t dim) {
    if (should_record(out, {&x})) {
        set_history(out, "UnsqueezeBackward", {x},
                    {[dim](const Tensor& g) { return g.squeeze(dim); }});
    }
}

void record_expand(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        set_history(out, "ExpandBackward", {x},
                    {[shape = x.shape()](const Tensor& g) { return sum_to_shape(g, shape); }});
    }
}

void record_slice(const Tensor& out, const Tensor& x, std::int64_t dim, std::int64_t start,
                  std::int64_t end, std::int64_t step) {
    if (should_record(out, {&x})) {
        set_history(
            out, "SliceBackward", {x},
            {[shape = x.shape(), dtype = x.dtype(), dim, start, end, step](const Tensor& g) {
                Tensor full_gradient = zeros(shape, dtype);
                full_gradient.slice(dim, start, end, step).copy_(g);
                return full_gradient;
            }});
    }
}

void record_select(const Tensor& out, const Tensor& x, std::int64_t dim, std::int64_t index) {
    if (should_record(out, {&x})) {
        set_history(out, "SelectBackward", {x},
                    {[shape = x.shape(), dtype = x.dtype(), dim, index](const Tensor& g) {
                        Tensor full_gradient = zeros(shape, dtype);
                        full_gradient.select(dim, index).copy_(g);
                        return full_gradient;
                    }});
    }
}

void record_clone(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        set_history(out, "CloneBackward", {x}, {[](const Tensor& g) { return g; }});
    }
}

void record_to(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        set_history(out, "ToBackward", {x},
                    {[dtype = x.dtype()](const Tensor& g) { return g.to(dtype); }});
    }
}

} // namespace kael::autograd
