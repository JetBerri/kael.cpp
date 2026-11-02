#include <algorithm>
#include <utility>

#include "internal/checked_math.h"
#include "kael/tensor.h"

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

std::size_t wrapped_index(std::int64_t dim, std::int64_t ndim) {
    return static_cast<std::size_t>(wrap_dim(dim, ndim));
}

template <typename Container>
void erase_at(Container& container, std::size_t position) {
    container.erase(container.begin() + static_cast<std::ptrdiff_t>(position));
}

template <typename Container>
void insert_at(Container& container, std::size_t position, std::int64_t value) {
    container.insert(container.begin() + static_cast<std::ptrdiff_t>(position), value);
}

Shape resolve_expand_shape(const Shape& current, const Shape& requested) {
    KAEL_CHECK(requested.size() >= current.size(), "cannot expand shape "
                                                       << to_string(current) << " to "
                                                       << to_string(requested)
                                                       << ": too few dimensions");
    const std::size_t new_dims = requested.size() - current.size();
    Shape resolved = requested;
    for (std::size_t i = 0; i < requested.size(); ++i) {
        if (requested[i] != -1) {
            continue;
        }
        KAEL_CHECK(i >= new_dims, "expand cannot use -1 for new dimension " << i << " in shape "
                                                                            << to_string(requested));
        resolved[i] = current[i - new_dims];
    }
    return resolved;
}

std::int64_t clamp_slice_bound(std::int64_t bound, std::int64_t size) {
    return std::clamp(bound < 0 ? bound + size : bound, std::int64_t{0}, size);
}

} // namespace

Tensor Tensor::view(const Shape& requested) const {
    KAEL_CHECK(is_contiguous(), "view needs a contiguous tensor, but shape "
                                    << to_string(shape()) << " has strides "
                                    << to_string(strides()) << "; use reshape instead");
    Shape target = infer_shape(requested, numel());
    Strides target_strides = contiguous_strides(target);
    return make_view(*this, std::move(target), std::move(target_strides), offset());
}

Tensor Tensor::reshape(const Shape& requested) const {
    return contiguous().view(requested);
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    const std::size_t first = wrapped_index(dim0, dim());
    const std::size_t second = wrapped_index(dim1, dim());

    Shape new_shape = shape();
    Strides new_strides = strides();
    std::swap(new_shape[first], new_shape[second]);
    std::swap(new_strides[first], new_strides[second]);
    return make_view(*this, std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::permute(const Dims& dims) const {
    KAEL_CHECK(static_cast<std::int64_t>(dims.size()) == dim(),
               "permute got " << dims.size() << " dimensions for a tensor with " << dim());

    std::vector<bool> seen(dims.size(), false);
    Shape new_shape(dims.size());
    Strides new_strides(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::size_t source = wrapped_index(dims[i], dim());
        KAEL_CHECK(!seen[source], "permute got dimension " << dims[i] << " more than once");
        seen[source] = true;
        new_shape[i] = shape()[source];
        new_strides[i] = strides()[source];
    }
    return make_view(*this, std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::T() const {
    KAEL_CHECK(dim() <= 2, "T needs at most 2 dimensions, got shape "
                               << to_string(shape()) << "; use transpose or mT instead");
    return dim() == 2 ? transpose(0, 1) : *this;
}

Tensor Tensor::mT() const {
    KAEL_CHECK(dim() >= 2, "mT needs at least 2 dimensions, got shape " << to_string(shape()));
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
    return make_view(*this, std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::squeeze(std::int64_t dim) const {
    const std::size_t position = wrapped_index(dim, this->dim());
    if (shape()[position] != 1) {
        return *this;
    }
    Shape new_shape = shape();
    Strides new_strides = strides();
    erase_at(new_shape, position);
    erase_at(new_strides, position);
    return make_view(*this, std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::unsqueeze(std::int64_t dim) const {
    const std::size_t position = wrapped_index(dim, this->dim() + 1);
    const std::int64_t new_stride =
        position < shape().size() ? shape()[position] * strides()[position] : 1;

    Shape new_shape = shape();
    Strides new_strides = strides();
    insert_at(new_shape, position, 1);
    insert_at(new_strides, position, new_stride);
    return make_view(*this, std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::expand(const Shape& requested) const {
    Shape target = resolve_expand_shape(shape(), requested);
    Strides target_strides = broadcast_strides(shape(), strides(), target);
    return make_view(*this, std::move(target), std::move(target_strides), offset());
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
    KAEL_CHECK(step > 0, "slice step must be positive, got " << step);

    const std::size_t position = wrapped_index(dim, this->dim());
    const std::int64_t first = clamp_slice_bound(start, shape()[position]);
    const std::int64_t last = clamp_slice_bound(end, shape()[position]);
    const std::int64_t stride = strides()[position];
    const std::int64_t new_offset =
        detail::checked_add(offset(), detail::checked_mul(first, stride, "slice offset"),
                            "slice offset");

    Shape new_shape = shape();
    Strides new_strides = strides();
    new_shape[position] = last > first ? (last - first - 1) / step + 1 : 0;
    new_strides[position] = detail::checked_mul(stride, step, "slice stride");
    return make_view(*this, std::move(new_shape), std::move(new_strides), new_offset);
}

Tensor Tensor::select(std::int64_t dim, std::int64_t index) const {
    const std::size_t position = wrapped_index(dim, this->dim());
    const std::int64_t size = shape()[position];
    const std::int64_t wrapped = index < 0 ? index + size : index;
    KAEL_CHECK(wrapped >= 0 && wrapped < size, "select index " << index << " out of range for size "
                                                               << size << " at dimension " << dim);

    Shape new_shape = shape();
    Strides new_strides = strides();
    const std::int64_t new_offset = offset() + wrapped * strides()[position];
    erase_at(new_shape, position);
    erase_at(new_strides, position);
    return make_view(*this, std::move(new_shape), std::move(new_strides), new_offset);
}

Tensor Tensor::narrow(std::int64_t dim, std::int64_t start, std::int64_t length) const {
    const std::int64_t size = shape()[wrapped_index(dim, this->dim())];
    const std::int64_t first = start < 0 ? start + size : start;
    KAEL_CHECK(first >= 0 && first <= size, "narrow start " << start << " out of range for size "
                                                            << size << " at dimension " << dim);
    KAEL_CHECK(length >= 0 && length <= size - first,
               "narrow length " << length << " does not fit size " << size << " from start "
                                << first);
    return slice(dim, first, first + length);
}

Tensor Tensor::flatten(std::int64_t start_dim, std::int64_t end_dim) const {
    if (dim() == 0) {
        return reshape({1});
    }
    const std::size_t first = wrapped_index(start_dim, dim());
    const std::size_t last = wrapped_index(end_dim, dim());
    KAEL_CHECK(first <= last,
               "flatten start dimension " << start_dim << " is after end dimension " << end_dim);
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
    const std::size_t position = wrapped_index(dim, this->dim());
    const Shape resolved = infer_shape(sizes, shape()[position]);

    const auto position_it = shape().begin() + static_cast<std::ptrdiff_t>(position);
    Shape target(shape().begin(), position_it);
    target.insert(target.end(), resolved.begin(), resolved.end());
    target.insert(target.end(), position_it + 1, shape().end());
    return reshape(target);
}

std::vector<Tensor> Tensor::split(std::int64_t chunk_size, std::int64_t dim) const {
    KAEL_CHECK(chunk_size > 0, "split size must be positive, got " << chunk_size);

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
    KAEL_CHECK(chunks > 0, "chunk count must be positive, got " << chunks);

    const std::int64_t total = size(dim);
    if (total == 0) {
        return std::vector<Tensor>(static_cast<std::size_t>(chunks), narrow(dim, 0, 0));
    }
    return split((total + chunks - 1) / chunks, dim);
}

} // namespace kael
