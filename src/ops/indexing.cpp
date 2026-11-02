#include <utility>

#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "internal/operands.h"

namespace kael {

namespace {

void check_same_except(const Tensor& reference, const Tensor& other, std::size_t position,
                       const char* op) {
    KAEL_CHECK(reference.dim() == other.dim(),
               op << " needs the same number of dimensions, got shapes "
                  << to_string(reference.shape()) << " and " << to_string(other.shape()));
    for (std::size_t i = 0; i < reference.shape().size(); ++i) {
        KAEL_CHECK(i == position || reference.shape()[i] == other.shape()[i],
                   op << " shapes " << to_string(reference.shape()) << " and "
                      << to_string(other.shape()) << " differ outside dimension " << position);
    }
}

void check_index_tensor(const Tensor& indices, const char* op) {
    KAEL_CHECK(indices.dtype() == DType::I64,
               op << " needs int64 indices, got " << indices.dtype());
}

std::int64_t checked_index(std::int64_t index, std::int64_t size, const char* op) {
    KAEL_CHECK(index >= 0 && index < size,
               op << " index " << index << " out of range [0, " << size << ")");
    return index;
}

void check_gather_shapes(const Tensor& t, const Tensor& indices, std::size_t position,
                         const char* op) {
    KAEL_CHECK(t.dim() == indices.dim(), op << " needs indices with the same number of dimensions,"
                                            << " got shapes " << to_string(t.shape()) << " and "
                                            << to_string(indices.shape()));
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
        KAEL_CHECK(i == position || indices.shape()[i] <= t.shape()[i],
                   op << " indices shape " << to_string(indices.shape())
                      << " is larger than tensor shape " << to_string(t.shape()) << " at dimension "
                      << i);
    }
}

template <typename Visit> void for_each_index(const Shape& shape, Visit visit) {
    const std::int64_t count = numel(shape);
    Shape index(shape.size(), 0);
    for (std::int64_t step = 0; step < count; ++step) {
        visit(index);
        for (std::size_t dim = shape.size(); dim-- > 0;) {
            if (++index[dim] < shape[dim]) {
                break;
            }
            index[dim] = 0;
        }
    }
}

} // namespace

Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dim) {
    KAEL_CHECK(!tensors.empty(), "cat needs at least one tensor");
    const Tensor& first = tensors.front();
    const auto position = static_cast<std::size_t>(wrap_dim(dim, first.dim()));

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

    std::int64_t start = 0;
    for (const Tensor& t : tensors) {
        const std::int64_t length = t.shape()[position];
        out.narrow(dim, start, length).copy_(t);
        start += length;
    }
    return out;
}

Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dim) {
    KAEL_CHECK(!tensors.empty(), "stack needs at least one tensor");
    const std::int64_t position = wrap_dim(dim, tensors.front().dim() + 1);

    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        KAEL_CHECK(t.shape() == tensors.front().shape(), "stack needs the same shape, got "
                                                             << to_string(tensors.front().shape())
                                                             << " and " << to_string(t.shape()));
        unsqueezed.push_back(t.unsqueeze(position));
    }
    return cat(unsqueezed, position);
}

Tensor where(const Tensor& condition, const Operand& a, const Operand& b) {
    KAEL_CHECK(condition.dtype() == DType::Bool,
               "where needs a bool condition, got " << condition.dtype());
    const auto [x, y] = detail::resolve_operands(a, b);
    return ternary_kernel(condition, x, y,
                          []<typename T>(T keep, T p, T q) -> T { return keep != T{0} ? p : q; });
}

Tensor masked_fill(const Tensor& t, const Tensor& mask, Scalar value) {
    Tensor out = t.clone();
    out.masked_fill_(mask, value);
    return out;
}

Tensor& Tensor::masked_fill_(const Tensor& mask, Scalar value) {
    KAEL_CHECK(mask.dtype() == DType::Bool, "masked_fill needs a bool mask, got " << mask.dtype());
    KAEL_CHECK(!has_internal_overlap(), "masked_fill_ cannot write to shape "
                                            << to_string(shape()) << " with strides "
                                            << to_string(strides())
                                            << " because several positions share one element;"
                                            << " call clone() first");
    binary_kernel_out(*this, *this, mask,
                      [fill = value.value()]<typename T>(T current, T selected) -> T {
                          return selected != T{0} ? convert_value<T>(fill) : current;
                      });
    return *this;
}

Tensor index_select(const Tensor& t, std::int64_t dim, const Tensor& indices) {
    check_index_tensor(indices, "index_select");
    KAEL_CHECK(indices.dim() == 1,
               "index_select needs 1-d indices, got shape " << to_string(indices.shape()));
    const auto position = static_cast<std::size_t>(wrap_dim(dim, t.dim()));

    Shape out_shape = t.shape();
    out_shape[position] = indices.numel();
    Tensor out = empty(out_shape, t.dtype());
    for (std::int64_t i = 0; i < indices.numel(); ++i) {
        const std::int64_t source =
            checked_index(indices.at<std::int64_t>({i}), t.shape()[position], "index_select");
        out.narrow(dim, i, 1).copy_(t.narrow(dim, source, 1));
    }
    return out;
}

Tensor gather(const Tensor& t, std::int64_t dim, const Tensor& indices) {
    check_index_tensor(indices, "gather");
    const auto position = static_cast<std::size_t>(wrap_dim(dim, t.dim()));
    check_gather_shapes(t, indices, position, "gather");

    Tensor out = empty(indices.shape(), t.dtype());
    dispatch(t.dtype(), [&]<typename T>() {
        for_each_index(indices.shape(), [&](const Shape& index) {
            Shape source = index;
            source[position] =
                checked_index(indices.at<std::int64_t>(index), t.shape()[position], "gather");
            out.at<T>(index) = t.at<T>(source);
        });
    });
    return out;
}

Tensor& Tensor::scatter_add_(std::int64_t dim, const Tensor& indices, const Tensor& source) {
    check_index_tensor(indices, "scatter_add_");
    const auto position = static_cast<std::size_t>(wrap_dim(dim, this->dim()));
    check_gather_shapes(*this, indices, position, "scatter_add_");
    KAEL_CHECK(source.dim() == indices.dim(),
               "scatter_add_ source shape " << to_string(source.shape()) << " and indices shape "
                                            << to_string(indices.shape())
                                            << " have different number of dimensions");
    for (std::size_t i = 0; i < indices.shape().size(); ++i) {
        KAEL_CHECK(indices.shape()[i] <= source.shape()[i], "scatter_add_ indices shape "
                                                                << to_string(indices.shape())
                                                                << " is larger than source shape "
                                                                << to_string(source.shape()));
    }

    const Tensor converted = source.to(dtype());
    dispatch(dtype(), [&]<typename T>() {
        for_each_index(indices.shape(), [&](const Shape& index) {
            Shape target = index;
            target[position] =
                checked_index(indices.at<std::int64_t>(index), shape()[position], "scatter_add_");
            T& element = at<T>(target);
            element = static_cast<T>(element + converted.at<T>(index));
        });
    });
    return *this;
}

} // namespace kael
