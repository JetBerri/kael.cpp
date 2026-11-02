#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <span>

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

    // ---- element access --------------------------------------------------------

    // The value of a tensor with exactly one element, whatever its number of
    // dimensions ([], [1] and [1, 1] all work). Throws otherwise, or if T
    // does not match the dtype.
    template <typename T>
    [[nodiscard]] T item() const {
        check_dtype<T>();
        check_single_element();
        return *reinterpret_cast<const T*>(data_ptr_raw());
    }

    // Same, converted to double from whatever the dtype is.
    [[nodiscard]] double item() const;

    // Reference to one element. Accepts Python-style negative indices (-1 is
    // the last one in that dimension). Throws if out of range, if the number
    // of indices does not match dim(), or if T does not match the dtype.
    template <typename T>
    [[nodiscard]] T& at(std::span<const std::int64_t> index) {
        check_dtype<T>();
        return *reinterpret_cast<T*>(impl_->storage->data() +
                                     element_offset(index) * element_size(dtype()));
    }

    template <typename T>
    [[nodiscard]] const T& at(std::span<const std::int64_t> index) const {
        check_dtype<T>();
        const Storage& storage = *impl_->storage;
        return *reinterpret_cast<const T*>(storage.data() +
                                           element_offset(index) * element_size(dtype()));
    }

    template <typename T>
    [[nodiscard]] T& at(std::initializer_list<std::int64_t> index) {
        return at<T>(std::span<const std::int64_t>(index.begin(), index.size()));
    }

    template <typename T>
    [[nodiscard]] const T& at(std::initializer_list<std::int64_t> index) const {
        return at<T>(std::span<const std::int64_t>(index.begin(), index.size()));
    }

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

    void check_single_element() const;

    // Element offset inside the storage of a (possibly negative) index.
    [[nodiscard]] std::int64_t element_offset(std::span<const std::int64_t> index) const;

    std::shared_ptr<TensorImpl> impl_;
};

// One line with the values in logical order, the shape and the dtype:
//   tensor([0.0000, 1.0000], shape=[2], dtype=float32)
// Tensors over 1000 elements only show the first and last 3 values.
std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace kael
