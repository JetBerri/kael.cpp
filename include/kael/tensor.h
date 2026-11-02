#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <span>
#include <vector>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/scalar.h"
#include "kael/shape.h"
#include "kael/tensor_impl.h"

namespace kael {

class Operand;

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
    [[nodiscard]] Tensor transpose(std::int64_t dim0, std::int64_t dim1) const;
    [[nodiscard]] Tensor permute(const std::vector<std::int64_t>& dims) const;
    [[nodiscard]] Tensor T() const;
    [[nodiscard]] Tensor mT() const;
    [[nodiscard]] Tensor squeeze() const;
    [[nodiscard]] Tensor squeeze(std::int64_t dim) const;
    [[nodiscard]] Tensor unsqueeze(std::int64_t dim) const;
    // The result reads several logical positions from one element: do not write to it.
    [[nodiscard]] Tensor expand(const Shape& shape) const;
    [[nodiscard]] bool has_internal_overlap() const;
    [[nodiscard]] Tensor slice(std::int64_t dim, std::int64_t start, std::int64_t end,
                               std::int64_t step = 1) const;
    [[nodiscard]] Tensor select(std::int64_t dim, std::int64_t index) const;
    [[nodiscard]] Tensor narrow(std::int64_t dim, std::int64_t start, std::int64_t length) const;
    [[nodiscard]] Tensor flatten(std::int64_t start_dim = 0, std::int64_t end_dim = -1) const;
    [[nodiscard]] Tensor unflatten(std::int64_t dim, const Shape& sizes) const;
    [[nodiscard]] std::vector<Tensor> split(std::int64_t chunk_size, std::int64_t dim = 0) const;
    [[nodiscard]] std::vector<Tensor> chunk(std::int64_t chunks, std::int64_t dim = 0) const;

    Tensor& add_(const Operand& other);
    Tensor& sub_(const Operand& other);
    Tensor& mul_(const Operand& other);
    Tensor& div_(const Operand& other);
    Tensor& pow_(const Operand& exponent);
    Tensor& clamp_(Scalar min, Scalar max);
    Tensor& copy_(const Tensor& source);
    Tensor& fill_(Scalar value);
    Tensor& zero_();

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
