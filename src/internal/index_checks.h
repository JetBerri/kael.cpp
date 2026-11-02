#pragma once

#include <cstdint>

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
