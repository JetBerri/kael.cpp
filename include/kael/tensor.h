#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/shape.h"
#include "kael/tensor_impl.h"

namespace kael {

// A tensor: a lightweight handle around a shared TensorImpl (docs/adr/0002).
//
// Copying a Tensor copies the pointer, not the data:
//
//   Tensor b = a;          // b and a are the same tensor
//   Tensor c = a.clone();  // c is an independent copy
//
// There is no default constructor: a Tensor always refers to valid data.
// Create tensors with the factories in kael/factory.h.
class Tensor {
public:
    // Internal entry point for factories, views and operations. Validates
    // the impl. Not meant for users of the library.
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    // ---- queries ----------------------------------------------------------

    [[nodiscard]] const Shape& shape() const noexcept { return impl_->shape; }
    [[nodiscard]] const Strides& strides() const noexcept { return impl_->strides; }
    [[nodiscard]] std::int64_t offset() const noexcept { return impl_->offset; }
    [[nodiscard]] DType dtype() const noexcept { return impl_->dtype; }
    [[nodiscard]] Device device() const noexcept { return impl_->device; }
    [[nodiscard]] std::int64_t numel() const { return impl_->numel(); }
    [[nodiscard]] std::int64_t dim() const { return impl_->dim(); }
    [[nodiscard]] bool is_contiguous() const { return impl_->is_contiguous(); }

    // Size and stride of one dimension; negative dims count from the end.
    [[nodiscard]] std::int64_t size(std::int64_t dim) const;
    [[nodiscard]] std::int64_t stride(std::int64_t dim) const;

    // numel * element_size: bytes of the elements, not of the storage.
    [[nodiscard]] std::int64_t nbytes() const;

    // Internal: the metadata behind the handle.
    [[nodiscard]] const std::shared_ptr<TensorImpl>& impl() const noexcept { return impl_; }

    // ---- raw data access ----------------------------------------------------

    // Pointer to the first element of the tensor (storage start + offset).
    // Throws if T does not match the dtype. For an empty tensor the pointer
    // must not be dereferenced (it may be null).
    template <typename T>
    [[nodiscard]] T* data_ptr() {
        check_dtype<T>();
        return reinterpret_cast<T*>(data_ptr_raw());
    }

    template <typename T>
    [[nodiscard]] const T* data_ptr() const {
        check_dtype<T>();
        return reinterpret_cast<const T*>(data_ptr_raw());
    }

    // Untyped pointer to the first element, for code that works on bytes.
    [[nodiscard]] std::byte* data_ptr_raw();
    [[nodiscard]] const std::byte* data_ptr_raw() const;

    // ---- copies and identity --------------------------------------------------

    // A new, contiguous tensor with the same values that shares nothing.
    [[nodiscard]] Tensor clone() const;

    // True if both handles point to the same impl (`b = a`).
    [[nodiscard]] bool is_same(const Tensor& other) const noexcept {
        return impl_ == other.impl_;
    }

    // True if both tensors read the same storage (e.g. two views).
    [[nodiscard]] bool shares_storage(const Tensor& other) const noexcept {
        return impl_->storage == other.impl_->storage;
    }

private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    template <typename T>
    void check_dtype() const {
        KAEL_CHECK(dtype_of_v<T> == dtype(),
                   "expected dtype " << dtype_of_v<T> << " but tensor has " << dtype());
    }

    std::shared_ptr<TensorImpl> impl_;
};

} // namespace kael
