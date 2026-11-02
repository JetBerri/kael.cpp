#pragma once

#include <utility>

#include "kael/factory.h"
#include "kael/tensor.h"

namespace kael::bench {

inline Tensor naive_add(const Tensor& a, const Tensor& b) {
    const Shape shape = broadcast_shapes(a.shape(), b.shape());
    const Strides strides_a = broadcast_strides(a.shape(), a.strides(), shape);
    const Strides strides_b = broadcast_strides(b.shape(), b.strides(), shape);
    const float* data_a = reinterpret_cast<const float*>(std::as_const(*a.impl()->storage).data());
    const float* data_b = reinterpret_cast<const float*>(std::as_const(*b.impl()->storage).data());

    Tensor out = empty(shape, DType::F32);
    float* data_out = out.data_ptr<float>();
    for (std::int64_t linear = 0; linear < out.numel(); ++linear) {
        const Shape index = index_from_linear(linear, shape);
        const std::int64_t offset_a = offset_from_index(shape, strides_a, a.offset(), index);
        const std::int64_t offset_b = offset_from_index(shape, strides_b, b.offset(), index);
        data_out[linear] = data_a[offset_a] + data_b[offset_b];
    }
    return out;
}

} // namespace kael::bench
