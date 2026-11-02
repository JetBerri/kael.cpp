#pragma once

#include <cstdint>
#include <vector>

#include "kael/device_kernels.h"
#include "kael/ops.h"
#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael::detail {

inline void check_index_tensor(const Tensor& indices, const char* op) {
    KAEL_CHECK_TYPE(indices.dtype() == DType::I64,
                    op << " needs int64 indices, got " << indices.dtype());
}

inline std::int64_t checked_index(std::int64_t index, std::int64_t size, const char* op) {
    KAEL_CHECK_INDEX(index >= 0 && index < size,
                     op << " index " << index << " out of range [0, " << size << ")");
    return index;
}

inline void check_gather_shapes(const Tensor& t, const Tensor& indices, std::size_t position,
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

inline void check_device_indices(const Tensor& indices, std::int64_t size, const char* op) {
    if (indices.numel() == 0) {
        return;
    }
    checked_index(kael::max(indices).item<std::int64_t>(), size, op);
    checked_index(kael::min(indices).item<std::int64_t>(), size, op);
}

inline device::IndexLayout index_layout(const Tensor& data, const Tensor& indices,
                                        const Tensor& source, std::size_t axis) {
    KAEL_CHECK(indices.dim() <= device::kMaxDims,
               "device kernels support at most " << device::kMaxDims << " dimensions");
    device::IndexLayout layout{};
    layout.dims = static_cast<int>(indices.dim());
    layout.axis = static_cast<int>(axis);
    layout.count = indices.numel();
    for (std::size_t dim = 0; dim < indices.shape().size(); ++dim) {
        layout.shape[dim] = indices.shape()[dim];
        layout.data_strides[dim] = data.strides()[dim];
        layout.index_strides[dim] = indices.strides()[dim];
        layout.source_strides[dim] = source.strides()[dim];
    }
    return layout;
}

class IndexWalker {
public:
    IndexWalker(const Tensor& target, const Tensor& indices, const Strides& source_strides,
                std::size_t axis)
        : shape_(indices.shape()), target_strides_(target.strides()),
          index_strides_(indices.strides()), source_strides_(source_strides), axis_(axis),
          counter_(shape_.size(), 0) {}

    [[nodiscard]] std::int64_t target_offset(std::int64_t index) const {
        return target_ + index * target_strides_[axis_];
    }
    [[nodiscard]] std::int64_t index_offset() const noexcept { return index_; }
    [[nodiscard]] std::int64_t source_offset() const noexcept { return source_; }

    void advance() noexcept {
        for (std::size_t dim = shape_.size(); dim-- > 0;) {
            ++counter_[dim];
            move(dim, 1);
            if (counter_[dim] < shape_[dim]) {
                return;
            }
            move(dim, -shape_[dim]);
            counter_[dim] = 0;
        }
    }

private:
    void move(std::size_t dim, std::int64_t steps) noexcept {
        index_ += steps * index_strides_[dim];
        source_ += steps * source_strides_[dim];
        if (dim != axis_) {
            target_ += steps * target_strides_[dim];
        }
    }

    const Shape& shape_;
    const Strides& target_strides_;
    const Strides& index_strides_;
    const Strides& source_strides_;
    std::size_t axis_;
    std::vector<std::int64_t> counter_;
    std::int64_t target_ = 0;
    std::int64_t index_ = 0;
    std::int64_t source_ = 0;
};

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

} // namespace kael::detail
