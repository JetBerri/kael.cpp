#include <utility>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "autograd/record.h"
#include "internal/index_checks.h"
#include "internal/operands.h"

namespace kael {

namespace {

void check_same_except(const Tensor& reference, const Tensor& other, std::size_t position,
                       const char* op) {
    KAEL_CHECK_VALUE(reference.dim() == other.dim(),
                     op << " needs the same number of dimensions, got shapes "
                        << to_string(reference.shape()) << " and " << to_string(other.shape()));
    for (std::size_t i = 0; i < reference.shape().size(); ++i) {
        KAEL_CHECK_VALUE(i == position || reference.shape()[i] == other.shape()[i],
                         op << " shapes " << to_string(reference.shape()) << " and "
                            << to_string(other.shape()) << " differ outside dimension "
                            << position);
    }
}

} // namespace

Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dim) {
    KAEL_CHECK_VALUE(!tensors.empty(), "cat needs at least one tensor");
    const Tensor& first = tensors.front();
    const std::size_t position = dim_position(dim, first.dim());

    DType dtype = first.dtype();
    std::int64_t total = 0;
    for (const Tensor& t : tensors) {
        check_same_except(first, t, position, "cat");
        dtype = promote_types(dtype, t.dtype());
        total += t.shape()[position];
    }

    Shape out_shape = first.shape();
    out_shape[position] = total;
    Tensor out = empty(out_shape, dtype);

    {
        const autograd::NoGradGuard guard;
        std::int64_t start = 0;
        for (const Tensor& t : tensors) {
            const std::int64_t length = t.shape()[position];
            out.narrow(dim, start, length).copy_(t);
            start += length;
        }
    }
    autograd::record_cat(out, tensors, dim);
    return out;
}

Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dim) {
    KAEL_CHECK_VALUE(!tensors.empty(), "stack needs at least one tensor");
    const std::int64_t position = wrap_dim(dim, tensors.front().dim() + 1);

    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        KAEL_CHECK_VALUE(t.shape() == tensors.front().shape(),
                         "stack needs the same shape, got " << to_string(tensors.front().shape())
                                                            << " and " << to_string(t.shape()));
        unsqueezed.push_back(t.unsqueeze(position));
    }
    return cat(unsqueezed, position);
}

Tensor where(const Tensor& condition, const Operand& a, const Operand& b) {
    KAEL_CHECK_TYPE(condition.dtype() == DType::Bool,
                    "where needs a bool condition, got " << condition.dtype());
    const auto [x, y] = detail::resolve_operands(a, b);
    Tensor out = ternary_kernel(
        condition, x, y, []<typename T>(T keep, T p, T q) -> T { return keep != T{0} ? p : q; });
    autograd::record_where(out, condition, x, y);
    return out;
}

Tensor masked_fill(const Tensor& t, const Tensor& mask, Scalar value) {
    Tensor out = autograd::without_grad([&] {
        Tensor filled = t.expand(broadcast_shapes(t.shape(), mask.shape())).clone();
        filled.masked_fill_(mask, value);
        return filled;
    });
    autograd::record_masked_fill(out, t, mask);
    return out;
}

Tensor index_select(const Tensor& t, std::int64_t dim, const Tensor& indices) {
    detail::check_index_tensor(indices, "index_select");
    KAEL_CHECK_VALUE(indices.dim() == 1,
                     "index_select needs 1-d indices, got shape " << to_string(indices.shape()));
    const std::size_t position = dim_position(dim, t.dim());

    Shape out_shape = t.shape();
    out_shape[position] = indices.numel();
    Tensor out = empty(out_shape, t.dtype());
    {
        const autograd::NoGradGuard guard;
        for (std::int64_t i = 0; i < indices.numel(); ++i) {
            const std::int64_t source = detail::checked_index(indices.at<std::int64_t>({i}),
                                                              t.shape()[position], "index_select");
            out.narrow(dim, i, 1).copy_(t.narrow(dim, source, 1));
        }
    }
    autograd::record_index_select(out, t, dim, indices);
    return out;
}

Tensor gather(const Tensor& t, std::int64_t dim, const Tensor& indices) {
    detail::check_index_tensor(indices, "gather");
    const std::size_t position = dim_position(dim, t.dim());
    detail::check_gather_shapes(t, indices, position, "gather");

    Tensor out = empty(indices.shape(), t.dtype());
    dispatch(t.dtype(), [&]<typename T>() {
        detail::for_each_index(indices.shape(), [&](const Shape& index) {
            Shape source = index;
            source[position] = detail::checked_index(indices.at<std::int64_t>(index),
                                                     t.shape()[position], "gather");
            out.at<T>(index) = t.at<T>(source);
        });
    });
    autograd::record_gather(out, t, dim, indices);
    return out;
}

} // namespace kael
