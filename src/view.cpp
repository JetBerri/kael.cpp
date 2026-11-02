#include <utility>

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

Tensor Tensor::permute(const std::vector<std::int64_t>& dims) const {
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

} // namespace kael
