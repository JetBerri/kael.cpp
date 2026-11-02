#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/shape.h"
#include "kael/tensor.h"

// Functions that create new tensors. Every result is contiguous, has offset
// 0 and owns a fresh storage. The default dtype is float32, as in PyTorch.

namespace kael {

// Allocates without initializing: the values are garbage until written.
Tensor empty(const Shape& shape, DType dtype = DType::F32);

Tensor zeros(const Shape& shape, DType dtype = DType::F32);
Tensor ones(const Shape& shape, DType dtype = DType::F32);

// Every element set to `value` converted to the dtype (for Bool, non-zero
// becomes true).
Tensor full(const Shape& shape, double value, DType dtype = DType::F32);

// Same shape and dtype as `t`.
Tensor empty_like(const Tensor& t);
Tensor zeros_like(const Tensor& t);
Tensor ones_like(const Tensor& t);

// 1-D tensor with start, start + step, ... up to `end` (excluded).
// Number of elements: ceil((end - start) / step), at least 0. Each value is
// computed as start + i * step (no accumulated rounding error).
// Throws if step is 0, if step goes the opposite way of [start, end), or for
// Bool.
Tensor arange(double end, DType dtype = DType::F32);
Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::F32);

// 1-D tensor of `steps` evenly spaced values from start to end, both
// included; the last value is exactly `end`. Floating point dtypes only.
Tensor linspace(double start, double end, std::int64_t steps, DType dtype = DType::F32);

// n x n identity matrix.
Tensor eye(std::int64_t n, DType dtype = DType::F32);

// 0-d tensor holding one value.
Tensor scalar(double value, DType dtype = DType::F32);

// Copies `data` into a new tensor of the given shape. The dtype comes from
// T (float -> F32, double -> F64, int64_t -> I64, uint8_t -> Bool). Beware of
// literals: {1.0, 2.0} is double (F64), {1.0f, 2.0f} is float (F32).
// Throws if data.size() != numel(shape).
template <typename T>
Tensor from_vector(const std::vector<T>& data, const Shape& shape) {
    const std::int64_t n = numel(shape);
    KAEL_CHECK(static_cast<std::int64_t>(data.size()) == n,
               "from_vector got " << data.size() << " values but shape " << to_string(shape)
                                  << " has " << n << " elements");
    Tensor t = empty(shape, dtype_of_v<T>);
    if (n > 0) {
        std::memcpy(t.data_ptr<T>(), data.data(), static_cast<std::size_t>(t.nbytes()));
    }
    return t;
}

// 1-D overload.
template <typename T>
Tensor from_vector(const std::vector<T>& data) {
    return from_vector(data, Shape{static_cast<std::int64_t>(data.size())});
}

// For tests and small examples: from_vector({1.f, 2.f, 3.f}, {3}).
template <typename T>
Tensor from_vector(std::initializer_list<T> data, const Shape& shape) {
    return from_vector(std::vector<T>(data), shape);
}

template <typename T>
Tensor from_vector(std::initializer_list<T> data) {
    return from_vector(std::vector<T>(data));
}

// std::vector<bool> stores packed bits, so it is converted element by
// element into the uint8_t representation of DType::Bool.
Tensor from_vector(const std::vector<bool>& data, const Shape& shape);
Tensor from_vector(const std::vector<bool>& data);
Tensor from_vector(std::initializer_list<bool> data, const Shape& shape);
Tensor from_vector(std::initializer_list<bool> data);

} // namespace kael
