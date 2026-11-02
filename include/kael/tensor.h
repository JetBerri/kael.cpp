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

class Tensor {
public:
    static Tensor from_impl(std::shared_ptr<TensorImpl> impl);

    [[nodiscard]] const Shape& shape() const noexcept { return impl_->shape; }
    [[nodiscard]] const Strides& strides() const noexcept { return impl_->strides; }
    [[nodiscard]] std::int64_t offset() const noexcept { return impl_->offset; }
    [[nodiscard]] DType dtype() const noexcept { return impl_->dtype; }
    [[nodiscard]] Device device() const noexcept { return impl_->device; }
    [[nodiscard]] std::int64_t numel() const { return impl_->numel(); }
    [[nodiscard]] std::int64_t dim() const { return impl_->dim(); }
    [[nodiscard]] bool is_contiguous() const { return impl_->is_contiguous(); }
    [[nodiscard]] std::int64_t size(std::int64_t dim) const;
    [[nodiscard]] std::int64_t stride(std::int64_t dim) const;
    [[nodiscard]] std::int64_t nbytes() const;
    [[nodiscard]] const std::shared_ptr<TensorImpl>& impl() const noexcept { return impl_; }

    template <typename T>
    [[nodiscard]] T* data_ptr() {
        check_dtype<T>();
        return reinterpret_cast<T*>(first_element());
    }

    template <typename T>
    [[nodiscard]] const T* data_ptr() const {
        check_dtype<T>();
        return reinterpret_cast<const T*>(first_element());
    }

    [[nodiscard]] std::byte* data_ptr_raw() { return first_element(); }
    [[nodiscard]] const std::byte* data_ptr_raw() const { return first_element(); }

    template <typename T>
    [[nodiscard]] T item() const {
        check_dtype<T>();
        check_single_element();
        return *reinterpret_cast<const T*>(first_element());
    }

    [[nodiscard]] double item() const;

    template <typename T>
    [[nodiscard]] T& at(std::span<const std::int64_t> index) {
        return *element_ptr<T>(index);
    }

    template <typename T>
    [[nodiscard]] const T& at(std::span<const std::int64_t> index) const {
        return *element_ptr<T>(index);
    }

    template <typename T>
    [[nodiscard]] T& at(std::initializer_list<std::int64_t> index) {
        return *element_ptr<T>({index.begin(), index.size()});
    }

    template <typename T>
    [[nodiscard]] const T& at(std::initializer_list<std::int64_t> index) const {
        return *element_ptr<T>({index.begin(), index.size()});
    }

    [[nodiscard]] Tensor to(DType dtype) const;
    [[nodiscard]] Tensor clone() const;
    [[nodiscard]] Tensor contiguous() const;

    [[nodiscard]] Tensor view(const Shape& shape) const;
    // Shares memory with this tensor only if it is contiguous; otherwise it copies.
    [[nodiscard]] Tensor reshape(const Shape& shape) const;

    [[nodiscard]] bool is_same(const Tensor& other) const noexcept {
        return impl_ == other.impl_;
    }

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

    template <typename T>
    [[nodiscard]] T* element_ptr(std::span<const std::int64_t> index) const {
        check_dtype<T>();
        return reinterpret_cast<T*>(impl_->storage->data()) + element_offset(index);
    }

    void check_single_element() const;
    [[nodiscard]] std::byte* first_element() const;
    [[nodiscard]] std::int64_t element_offset(std::span<const std::int64_t> index) const;

    std::shared_ptr<TensorImpl> impl_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace kael
