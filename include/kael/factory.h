#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael {

Tensor empty(const Shape& shape, DType dtype = DType::F32);
Tensor zeros(const Shape& shape, DType dtype = DType::F32);
Tensor ones(const Shape& shape, DType dtype = DType::F32);
Tensor full(const Shape& shape, double value, DType dtype = DType::F32);

Tensor empty_like(const Tensor& t);
Tensor zeros_like(const Tensor& t);
Tensor ones_like(const Tensor& t);

Tensor arange(double end, DType dtype = DType::F32);
Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::F32);
Tensor linspace(double start, double end, std::int64_t steps, DType dtype = DType::F32);
Tensor eye(std::int64_t n, DType dtype = DType::F32);
Tensor scalar(double value, DType dtype = DType::F32);

namespace detail {
void check_value_count(std::size_t value_count, const Shape& shape);
}

template <typename T>
Tensor from_vector(const std::vector<T>& data, const Shape& shape) {
    detail::check_value_count(data.size(), shape);
    Tensor t = empty(shape, dtype_of_v<T>);
    if (!data.empty()) {
        std::memcpy(t.data_ptr<T>(), data.data(), static_cast<std::size_t>(t.nbytes()));
    }
    return t;
}

template <typename T>
Tensor from_vector(const std::vector<T>& data) {
    return from_vector(data, Shape{static_cast<std::int64_t>(data.size())});
}

template <typename T>
Tensor from_vector(std::initializer_list<T> data, const Shape& shape) {
    return from_vector(std::vector<T>(data), shape);
}

template <typename T>
Tensor from_vector(std::initializer_list<T> data) {
    return from_vector(std::vector<T>(data));
}

Tensor from_vector(const std::vector<bool>& data, const Shape& shape);
Tensor from_vector(const std::vector<bool>& data);
Tensor from_vector(std::initializer_list<bool> data, const Shape& shape);
Tensor from_vector(std::initializer_list<bool> data);

} // namespace kael
