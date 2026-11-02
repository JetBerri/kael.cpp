#include <algorithm>
#include <utility>

#include "kael/tensor.h"

#include "autograd/record.h"
#include "internal/checked_math.h"

namespace kael {

namespace {

Tensor make_view(const Tensor& base, Shape shape, Strides strides, std::int64_t offset) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = base.impl()->storage;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = offset;
    impl->dtype = base.dtype();
    impl->device = base.device();
    return Tensor::from_impl(std::move(impl));
}

template <typename Container> void erase_at(Container& container, std::size_t position) {
    container.erase(container.begin() + static_cast<std::ptrdiff_t>(position));
}

template <typename Container>
void insert_at(Container& container, std::size_t position, std::int64_t value) {
    container.insert(container.begin() + static_cast<std::ptrdiff_t>(position), value);
}

Shape resolve_expand_shape(const Shape& current, const Shape& requested) {
    KAEL_CHECK_VALUE(requested.size() >= current.size(),
                     "cannot expand shape " << to_string(current) << " to " << to_string(requested)
                                            << ": too few dimensions");
    const std::size_t new_dims = requested.size() - current.size();
    Shape resolved = requested;
    for (std::size_t i = 0; i < requested.size(); ++i) {
        if (requested[i] != -1) {
            continue;
        }
        KAEL_CHECK_VALUE(i >= new_dims, "expand cannot use -1 for new dimension "
                                            << i << " in shape " << to_string(requested));
        resolved[i] = current[i - new_dims];
    }
    return resolved;
}

std::int64_t clamp_slice_bound(std::int64_t bound, std::int64_t size) {
    return std::clamp(bound < 0 ? bound + size : bound, std::int64_t{0}, size);
}

} // namespace

Tensor Tensor::view(const Shape& requested) const {
    KAEL_CHECK_VALUE(is_contiguous(), "view needs a contiguous tensor, but shape "
                                          << to_string(shape()) << " has strides "
                                          << to_string(strides()) << "; use reshape instead");
    Shape target = infer_shape(requested, numel());
    Strides target_strides = contiguous_strides(target);
    Tensor out = make_view(*this, std::move(target), std::move(target_strides), offset());
    autograd::record_view(out, *this);
    return out;
}

Tensor Tensor::detach() const {
    return make_view(*this, shape(), strides(), offset());
}

Tensor Tensor::reshape(const Shape& requested) const {
    return contiguous().view(requested);
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    if (dim() == 0) {
        wrap_dim(dim0, 1);
        wrap_dim(dim1, 1);
        return *this;
    }
    const std::size_t first = dim_position(dim0, dim());
    const std::size_t second = dim_position(dim1, dim());

    Shape new_shape = shape();
    Strides new_strides = strides();
    std::swap(new_shape[first], new_shape[second]);
    std::swap(new_strides[first], new_strides[second]);
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_transpose(out, *this, dim0, dim1);
    return out;
}

Tensor Tensor::permute(const Dims& dims) const {
    KAEL_CHECK_VALUE(static_cast<std::int64_t>(dims.size()) == dim(),
                     "permute got " << dims.size() << " dimensions for a tensor with " << dim());

    std::vector<bool> seen(dims.size(), false);
    Shape new_shape(dims.size());
    Strides new_strides(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::size_t source = dim_position(dims[i], dim());
        KAEL_CHECK_VALUE(!seen[source], "permute got dimension " << dims[i] << " more than once");
        seen[source] = true;
        new_shape[i] = shape()[source];
        new_strides[i] = strides()[source];
    }
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_permute(out, *this, dims);
    return out;
}

Tensor Tensor::T() const {
    KAEL_CHECK_VALUE(dim() <= 2, "T needs at most 2 dimensions, got shape "
                                     << to_string(shape()) << "; use transpose or mT instead");
    return dim() == 2 ? transpose(0, 1) : *this;
}

Tensor Tensor::mT() const {
    KAEL_CHECK_VALUE(dim() >= 2,
                     "mT needs at least 2 dimensions, got shape " << to_string(shape()));
    return transpose(-2, -1);
}

Tensor Tensor::squeeze() const {
    Shape new_shape;
    Strides new_strides;
    for (std::size_t i = 0; i < shape().size(); ++i) {
        if (shape()[i] != 1) {
            new_shape.push_back(shape()[i]);
            new_strides.push_back(strides()[i]);
        }
    }
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_squeeze(out, *this);
    return out;
}

Tensor Tensor::squeeze(std::int64_t dim) const {
    if (this->dim() == 0) {
        wrap_dim(dim, 1);
        return *this;
    }
    const std::size_t position = dim_position(dim, this->dim());
    if (shape()[position] != 1) {
        return *this;
    }
    Shape new_shape = shape();
    Strides new_strides = strides();
    erase_at(new_shape, position);
    erase_at(new_strides, position);
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_squeeze(out, *this, dim);
    return out;
}

Tensor Tensor::unsqueeze(std::int64_t dim) const {
    const std::size_t position = dim_position(dim, this->dim() + 1);
    const std::int64_t new_stride =
        position < shape().size() ? shape()[position] * strides()[position] : 1;

    Shape new_shape = shape();
    Strides new_strides = strides();
    insert_at(new_shape, position, 1);
    insert_at(new_strides, position, new_stride);
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), offset());
    autograd::record_unsqueeze(out, *this, dim);
    return out;
}

Tensor Tensor::expand(const Shape& requested) const {
    Shape target = resolve_expand_shape(shape(), requested);
    Strides target_strides = broadcast_strides(shape(), strides(), target);
    Tensor out = make_view(*this, std::move(target), std::move(target_strides), offset());
    autograd::record_expand(out, *this);
    return out;
}

bool Tensor::has_internal_overlap() const {
    for (std::size_t i = 0; i < shape().size(); ++i) {
        if (shape()[i] > 1 && strides()[i] == 0) {
            return true;
        }
    }
    return false;
}

Tensor Tensor::slice(std::int64_t dim, std::int64_t start, std::int64_t end,
                     std::int64_t step) const {
    KAEL_CHECK_VALUE(step > 0, "slice step must be positive, got " << step);

    const std::size_t position = dim_position(dim, this->dim());
    const std::int64_t first = clamp_slice_bound(start, shape()[position]);
    const std::int64_t last = clamp_slice_bound(end, shape()[position]);
    const std::int64_t stride = strides()[position];
    const std::int64_t new_offset = detail::checked_add(
        offset(), detail::checked_mul(first, stride, "slice offset"), "slice offset");

    Shape new_shape = shape();
    Strides new_strides = strides();
    new_shape[position] = last > first ? (last - first - 1) / step + 1 : 0;
    new_strides[position] = detail::checked_mul(stride, step, "slice stride");
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), new_offset);
    autograd::record_slice(out, *this, dim, start, end, step);
    return out;
}

Tensor Tensor::select(std::int64_t dim, std::int64_t index) const {
    const std::size_t position = dim_position(dim, this->dim());
    const std::int64_t size = shape()[position];
    const std::int64_t wrapped = index < 0 ? index + size : index;
    KAEL_CHECK_INDEX(wrapped >= 0 && wrapped < size, "select index "
                                                         << index << " out of range for size "
                                                         << size << " at dimension " << dim);

    Shape new_shape = shape();
    Strides new_strides = strides();
    const std::int64_t new_offset = offset() + wrapped * strides()[position];
    erase_at(new_shape, position);
    erase_at(new_strides, position);
    Tensor out = make_view(*this, std::move(new_shape), std::move(new_strides), new_offset);
    autograd::record_select(out, *this, dim, index);
    return out;
}

Tensor Tensor::narrow(std::int64_t dim, std::int64_t start, std::int64_t length) const {
    const std::int64_t size = shape()[dim_position(dim, this->dim())];
    const std::int64_t first = start < 0 ? start + size : start;
    KAEL_CHECK_INDEX(first >= 0 && first <= size, "narrow start "
                                                      << start << " out of range for size " << size
                                                      << " at dimension " << dim);
    KAEL_CHECK_VALUE(length >= 0 && length <= size - first, "narrow length "
                                                                << length << " does not fit size "
                                                                << size << " from start " << first);
    return slice(dim, first, first + length);
}

Tensor Tensor::flatten(std::int64_t start_dim, std::int64_t end_dim) const {
    if (dim() == 0) {
        return reshape({1});
    }
    const std::size_t first = dim_position(start_dim, dim());
    const std::size_t last = dim_position(end_dim, dim());
    KAEL_CHECK_VALUE(first <= last, "flatten start dimension "
                                        << start_dim << " is after end dimension " << end_dim);
    if (first == last) {
        return *this;
    }

    const auto begin = shape().begin();
    const auto first_it = begin + static_cast<std::ptrdiff_t>(first);
    const auto after_last_it = begin + static_cast<std::ptrdiff_t>(last) + 1;

    Shape target(begin, first_it);
    target.push_back(kael::numel(Shape(first_it, after_last_it)));
    target.insert(target.end(), after_last_it, shape().end());
    return reshape(target);
}

Tensor Tensor::unflatten(std::int64_t dim, const Shape& sizes) const {
    const std::size_t position = dim_position(dim, this->dim());
    const Shape resolved = infer_shape(sizes, shape()[position]);

    const auto position_it = shape().begin() + static_cast<std::ptrdiff_t>(position);
    Shape target(shape().begin(), position_it);
    target.insert(target.end(), resolved.begin(), resolved.end());
    target.insert(target.end(), position_it + 1, shape().end());
    return reshape(target);
}

std::vector<Tensor> Tensor::split(std::int64_t chunk_size, std::int64_t dim) const {
    KAEL_CHECK_VALUE(chunk_size > 0, "split size must be positive, got " << chunk_size);

    const std::int64_t total = size(dim);
    std::vector<Tensor> pieces;
    std::int64_t start = 0;
    do {
        pieces.push_back(narrow(dim, start, std::min(chunk_size, total - start)));
        start += chunk_size;
    } while (start < total);
    return pieces;
}

std::vector<Tensor> Tensor::chunk(std::int64_t chunks, std::int64_t dim) const {
    KAEL_CHECK_VALUE(chunks > 0, "chunk count must be positive, got " << chunks);

    const std::int64_t total = size(dim);
    if (total == 0) {
        return std::vector<Tensor>(static_cast<std::size_t>(chunks), narrow(dim, 0, 0));
    }
    return split((total + chunks - 1) / chunks, dim);
}

} // namespace kael
