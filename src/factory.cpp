#include "kael/factory.h"

#include <algorithm>
#include <cmath>

#include "internal/checked_math.h"

namespace kael {

namespace {

// Writes `value` (converted to the dtype) into every element of a freshly
// created, contiguous tensor.
void fill_contiguous(Tensor& t, double value) {
    dispatch(t.dtype(), [&]<typename T>() {
        std::fill_n(t.data_ptr<T>(), t.numel(), convert_value<T>(value));
    });
}

} // namespace

Tensor empty(const Shape& shape, DType dtype) {
    const std::int64_t n = numel(shape); // also validates the shape
    const std::int64_t bytes = detail::checked_mul(n, element_size(dtype), "tensor size in bytes");

    auto impl = std::make_shared<TensorImpl>();
    impl->storage = std::make_shared<Storage>(bytes, Device::CPU);
    impl->shape = shape;
    impl->strides = contiguous_strides(shape);
    impl->offset = 0;
    impl->dtype = dtype;
    impl->device = Device::CPU;
    return Tensor::from_impl(std::move(impl));
}

Tensor full(const Shape& shape, double value, DType dtype) {
    Tensor t = empty(shape, dtype);
    fill_contiguous(t, value);
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
    KAEL_CHECK(std::isfinite(start) && std::isfinite(end) && std::isfinite(step),
               "arange needs finite start, end and step, got " << start << ", " << end << ", "
                                                               << step);
    KAEL_CHECK(step != 0.0, "arange step must be non-zero");
    KAEL_CHECK((step > 0.0 && start <= end) || (step < 0.0 && start >= end),
               "arange step " << step << " goes the wrong way from " << start << " to " << end);

    const double count = std::ceil((end - start) / step);
    KAEL_CHECK(count < 9.0e18, "arange would produce too many elements (" << count << ")");
    const auto n = static_cast<std::int64_t>(std::max(count, 0.0));

    Tensor t = empty({n}, dtype);
    dispatch(dtype, [&]<typename T>() {
        T* out = t.data_ptr<T>();
        for (std::int64_t i = 0; i < n; ++i) {
            out[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    });
    return t;
}

Tensor linspace(double start, double end, std::int64_t steps, DType dtype) {
    KAEL_CHECK(steps >= 0, "linspace number of steps must be non-negative, got " << steps);
    Tensor t = empty({steps}, dtype);
    dispatch_floating(dtype, "linspace", [&]<typename T>() {
        T* out = t.data_ptr<T>();
        if (steps == 0) {
            return;
        }
        out[0] = static_cast<T>(start);
        if (steps == 1) {
            return;
        }
        const double delta = (end - start) / static_cast<double>(steps - 1);
        for (std::int64_t i = 1; i < steps - 1; ++i) {
            out[i] = static_cast<T>(start + static_cast<double>(i) * delta);
        }
        out[steps - 1] = static_cast<T>(end); // exact, whatever the rounding
    });
    return t;
}

Tensor eye(std::int64_t n, DType dtype) {
    KAEL_CHECK(n >= 0, "eye size must be non-negative, got " << n);
    Tensor t = zeros({n, n}, dtype);
    dispatch(dtype, [&]<typename T>() {
        T* out = t.data_ptr<T>();
        for (std::int64_t i = 0; i < n; ++i) {
            out[i * n + i] = convert_value<T>(1.0);
        }
    });
    return t;
}

Tensor scalar(double value, DType dtype) {
    return full({}, value, dtype);
}

Tensor from_vector(const std::vector<bool>& data, const Shape& shape) {
    const std::int64_t n = numel(shape);
    KAEL_CHECK(static_cast<std::int64_t>(data.size()) == n,
               "from_vector got " << data.size() << " values but shape " << to_string(shape)
                                  << " has " << n << " elements");
    Tensor t = empty(shape, DType::Bool);
    std::uint8_t* out = t.data_ptr<std::uint8_t>();
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = data[i] ? 1 : 0;
    }
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
