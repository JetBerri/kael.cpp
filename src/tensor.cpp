#include "kael/tensor.h"

#include <cstring>
#include <utility>

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

std::byte* Tensor::data_ptr_raw() {
    std::byte* base = impl_->storage->data();
    // An empty tensor may have a null storage, or an offset that points at
    // the very end of it; never do pointer arithmetic on those.
    if (base == nullptr || numel() == 0) {
        return base;
    }
    return base + offset() * element_size(dtype());
}

const std::byte* Tensor::data_ptr_raw() const {
    const std::byte* base = std::as_const(*impl_->storage).data();
    if (base == nullptr || numel() == 0) {
        return base;
    }
    return base + offset() * element_size(dtype());
}

Tensor Tensor::clone() const {
    // First version: contiguous tensors only. Generalized in phase 4.
    KAEL_CHECK(is_contiguous(), "clone of a non-contiguous tensor is not implemented yet");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(nbytes(), device());
    impl->shape = shape();
    impl->strides = contiguous_strides(shape());
    impl->offset = 0;
    impl->dtype = dtype();
    impl->device = device();

    if (nbytes() > 0) {
        std::memcpy(impl->storage->data(), data_ptr_raw(), static_cast<std::size_t>(nbytes()));
    }
    return Tensor(std::move(impl));
}

} // namespace kael
