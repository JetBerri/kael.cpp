#include "kael/factory.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "internal/checked_math.h"

namespace kael {

namespace {

constexpr double kMaxArangeElements = 9.0e18;

std::int64_t arange_count(double start, double end, double step) {
    KAEL_CHECK(std::isfinite(start) && std::isfinite(end) && std::isfinite(step),
               "arange needs finite start, end and step, got " << start << ", " << end << ", "
                                                               << step);
    KAEL_CHECK(step != 0.0, "arange step must be non-zero");
    KAEL_CHECK((step > 0.0 && start <= end) || (step < 0.0 && start >= end),
               "arange step " << step << " goes the wrong way from " << start << " to " << end);

    const double count = std::ceil((end - start) / step);
    KAEL_CHECK(count < kMaxArangeElements,
               "arange would produce too many elements (" << count << ")");
    return static_cast<std::int64_t>(count);
}

} // namespace

namespace detail {

void check_value_count(std::size_t value_count, const Shape& shape) {
    const std::int64_t expected = numel(shape);
    KAEL_CHECK(static_cast<std::int64_t>(value_count) == expected,
               "from_vector got " << value_count << " values but shape " << to_string(shape)
                                  << " has " << expected << " elements");
}

} // namespace detail

Tensor empty(const Shape& shape, DType dtype) {
    const std::int64_t bytes =
        detail::checked_mul(numel(shape), element_size(dtype), "tensor size in bytes");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(bytes, Device::CPU);
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    impl->dtype = dtype;
    return Tensor::from_impl(std::move(impl));
}

Tensor full(const Shape& shape, double value, DType dtype) {
    Tensor t = empty(shape, dtype);
    dispatch(dtype, [&]<typename T>() {
        std::fill_n(t.data_ptr<T>(), t.numel(), convert_value<T>(value));
    });
    return t;
}

Tensor zeros(const Shape& shape, DType dtype) {
    return full(shape, 0.0, dtype);
}

Tensor ones(const Shape& shape, DType dtype) {
    return full(shape, 1.0, dtype);
}

Tensor empty_like(const Tensor& t) {
    return empty(t.shape(), t.dtype());
}

Tensor zeros_like(const Tensor& t) {
    return zeros(t.shape(), t.dtype());
}

Tensor ones_like(const Tensor& t) {
    return ones(t.shape(), t.dtype());
}

Tensor arange(double end, DType dtype) {
    return arange(0.0, end, 1.0, dtype);
}

Tensor arange(double start, double end, double step, DType dtype) {
    KAEL_CHECK(dtype != DType::Bool, "arange does not support dtype bool");
    const std::int64_t count = arange_count(start, end, step);

    Tensor t = empty({count}, dtype);
    dispatch(dtype, [&]<typename T>() {
        T* values = t.data_ptr<T>();
        for (std::int64_t i = 0; i < count; ++i) {
            values[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    });
    return t;
}

Tensor linspace(double start, double end, std::int64_t steps, DType dtype) {
    KAEL_CHECK(steps >= 0, "linspace number of steps must be non-negative, got " << steps);

    Tensor t = empty({steps}, dtype);
    dispatch_floating(dtype, "linspace", [&]<typename T>() {
        T* values = t.data_ptr<T>();
        const double delta = steps > 1 ? (end - start) / static_cast<double>(steps - 1) : 0.0;
        for (std::int64_t i = 0; i < steps; ++i) {
            values[i] = static_cast<T>(start + static_cast<double>(i) * delta);
        }
        if (steps > 1) {
            values[steps - 1] = static_cast<T>(end);
        }
    });
    return t;
}

Tensor eye(std::int64_t n, DType dtype) {
    KAEL_CHECK(n >= 0, "eye size must be non-negative, got " << n);

    Tensor t = zeros({n, n}, dtype);
    dispatch(dtype, [&]<typename T>() {
        T* values = t.data_ptr<T>();
        for (std::int64_t i = 0; i < n; ++i) {
            values[i * n + i] = convert_value<T>(1.0);
        }
    });
    return t;
}

Tensor scalar(double value, DType dtype) {
    return full({}, value, dtype);
}

Tensor from_vector(const std::vector<bool>& data, const Shape& shape) {
    detail::check_value_count(data.size(), shape);
    Tensor t = empty(shape, DType::Bool);
    std::copy(data.begin(), data.end(), t.data_ptr<std::uint8_t>());
    return t;
}

Tensor from_vector(const std::vector<bool>& data) {
    return from_vector(data, Shape{static_cast<std::int64_t>(data.size())});
}

Tensor from_vector(std::initializer_list<bool> data, const Shape& shape) {
    return from_vector(std::vector<bool>(data), shape);
}

Tensor from_vector(std::initializer_list<bool> data) {
    return from_vector(std::vector<bool>(data));
}

} // namespace kael
