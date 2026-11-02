#include "kael/tensor.h"

#include <cstring>
#include <utility>
#include <vector>

#include "internal/copy.h"
#include "kael/factory.h"

namespace kael {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    KAEL_CHECK(impl_ != nullptr, "cannot create a tensor from a null impl");
    impl_->validate();
}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}

std::int64_t Tensor::size(std::int64_t dim) const {
    return shape()[static_cast<std::size_t>(wrap_dim(dim, this->dim()))];
}

std::int64_t Tensor::stride(std::int64_t dim) const {
    return strides()[static_cast<std::size_t>(wrap_dim(dim, this->dim()))];
}

std::int64_t Tensor::nbytes() const {
    return numel() * element_size(dtype());
}

std::byte* Tensor::first_element() const {
    std::byte* base = impl_->storage->data();
    // An empty view can have an offset past the end of its storage, or a null storage.
    return numel() == 0 ? base : base + offset() * element_size(dtype());
}

void Tensor::check_single_element() const {
    KAEL_CHECK(numel() == 1, "item needs a tensor with one element, got shape "
                                 << to_string(shape()) << " with " << numel() << " elements");
}

double Tensor::item() const {
    check_single_element();
    return dispatch(dtype(), [&]<typename T>() {
        return static_cast<double>(*reinterpret_cast<const T*>(first_element()));
    });
}

std::int64_t Tensor::element_offset(std::span<const std::int64_t> index) const {
    KAEL_CHECK(static_cast<std::int64_t>(index.size()) == dim(),
               "got " << index.size() << " indices for a tensor with " << dim() << " dimensions");
    std::vector<std::int64_t> wrapped(index.begin(), index.end());
    for (std::size_t i = 0; i < wrapped.size(); ++i) {
        if (wrapped[i] < 0) {
            wrapped[i] += shape()[i];
        }
    }
    return offset_from_index(shape(), strides(), offset(), wrapped);
}

Tensor Tensor::to(DType target) const {
    if (target == dtype()) {
        return *this;
    }
    Tensor out = empty(shape(), target);
    detail::convert_strided(out, *this);
    return out;
}

Tensor Tensor::clone() const {
    Tensor out = empty(shape(), dtype());
    if (!is_contiguous()) {
        detail::copy_strided(out, *this);
    } else if (nbytes() > 0) {
        std::memcpy(out.data_ptr_raw(), data_ptr_raw(), static_cast<std::size_t>(nbytes()));
    }
    return out;
}

Tensor Tensor::contiguous() const {
    return is_contiguous() ? *this : clone();
}

} // namespace kael
