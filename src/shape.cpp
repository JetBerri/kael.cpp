#include "kael/shape.h"

#include <algorithm>
#include <iterator>
#include <limits>

#include "kael/error.h"

namespace kael {

namespace {

constexpr std::int64_t kInferredDim = -1;

std::string join(std::span<const std::int64_t> values) {
    std::string text = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        text += std::to_string(values[i]);
    }
    return text + "]";
}

} // namespace

std::string to_string(const Shape& shape) {
    return join(shape);
}

void validate_shape(const Shape& shape) {
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK(shape[i] >= 0, "negative dimension: " << shape[i] << " at position " << i
                                                         << " in shape: " << to_string(shape));
    }
}

std::int64_t numel(const Shape& shape) {
    validate_shape(shape);

    if (std::ranges::find(shape, 0) != shape.end()) {
        return 0;
    }

    std::int64_t total = 1;
    for (const std::int64_t dim : shape) {
        KAEL_CHECK(total <= std::numeric_limits<std::int64_t>::max() / dim,
                   "number of elements overflows int64 for shape " << to_string(shape));
        total *= dim;
    }
    return total;
}

Strides contiguous_strides(const Shape& shape) {
    validate_shape(shape);

    Strides strides(shape.size());
    std::int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= std::max<std::int64_t>(shape[i], 1);
    }
    return strides;
}

std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                               std::span<const std::int64_t> strides, std::int64_t base_offset,
                               std::span<const std::int64_t> index) {
    KAEL_CHECK(shape.size() == strides.size(),
               "shape and strides have different number of dimensions: shape "
                   << join(shape) << " has " << shape.size() << ", strides " << join(strides)
                   << " has " << strides.size());
    KAEL_CHECK(index.size() == shape.size(),
               "index and shape have different number of dimensions: index "
                   << join(index) << " has " << index.size() << ", shape " << join(shape) << " has "
                   << shape.size());

    std::int64_t offset = base_offset;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK(index[i] >= 0 && index[i] < shape[i],
                   "index " << index[i] << " out of range [0, " << shape[i] << ") at dimension "
                            << i << " in shape " << join(shape));
        offset += index[i] * strides[i];
    }
    return offset;
}

Shape index_from_linear(std::int64_t linear, const Shape& shape) {
    const std::int64_t count = numel(shape);
    KAEL_CHECK(linear >= 0 && linear < count, "linear index " << linear << " out of range [0, "
                                                              << count << ") for shape "
                                                              << to_string(shape));

    Shape index(shape.size());
    for (std::size_t i = shape.size(); i-- > 0;) {
        index[i] = linear % shape[i];
        linear /= shape[i];
    }
    return index;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    validate_shape(a);
    validate_shape(b);

    const std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);
    for (std::size_t from_right = 0; from_right < ndim; ++from_right) {
        const std::int64_t dim_a = from_right < a.size() ? a[a.size() - 1 - from_right] : 1;
        const std::int64_t dim_b = from_right < b.size() ? b[b.size() - 1 - from_right] : 1;
        KAEL_CHECK(dim_a == dim_b || dim_a == 1 || dim_b == 1,
                   "shapes " << to_string(a) << " and " << to_string(b)
                             << " are not broadcastable");
        result[ndim - 1 - from_right] = dim_a == 1 ? dim_b : dim_a;
    }
    return result;
}

Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target) {
    validate_shape(shape);
    validate_shape(target);
    KAEL_CHECK(shape.size() == strides.size(),
               "shape " << to_string(shape) << " and strides " << to_string(strides)
                        << " do not match size " << shape.size() << ":" << strides.size());
    KAEL_CHECK(shape.size() <= target.size(),
               "shape size " << to_string(shape) << " is bigger than target shape"
                             << to_string(target) << " " << shape.size() << ":" << target.size());

    const std::size_t new_dims = target.size() - shape.size();
    Strides result(target.size(), 0);
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const std::int64_t target_dim = target[i + new_dims];
        KAEL_CHECK(shape[i] == target_dim || shape[i] == 1,
                   "incompatible shapes " << to_string(shape) << " and " << to_string(target));
        if (shape[i] == target_dim) {
            result[i + new_dims] = strides[i];
        }
    }
    return result;
}

std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim) {
    KAEL_CHECK(ndim > 0, "dimension " << dim << " specified but tensor has no dimensions");
    KAEL_CHECK(dim >= -ndim && dim < ndim,
               "dimension " << dim << " out of range [" << -ndim << ", " << ndim - 1 << "]");
    return dim < 0 ? dim + ndim : dim;
}

std::size_t dim_position(std::int64_t dim, std::int64_t ndim) {
    return static_cast<std::size_t>(wrap_dim(dim, ndim));
}

Shape infer_shape(const Shape& requested, std::int64_t element_count) {
    KAEL_CHECK(std::ranges::count(requested, kInferredDim) <= 1,
               "only one dimension can be -1 in shape " << to_string(requested));

    Shape known;
    std::ranges::copy_if(requested, std::back_inserter(known),
                         [](std::int64_t dim) { return dim != kInferredDim; });
    const std::int64_t known_count = numel(known);

    const auto inferred = std::ranges::find(requested, kInferredDim);
    if (inferred == requested.end()) {
        KAEL_CHECK(known_count == element_count,
                   "shape " << to_string(requested) << " has " << known_count
                            << " elements but the tensor has " << element_count);
        return requested;
    }

    KAEL_CHECK(known_count != 0, "cannot infer the -1 in shape "
                                     << to_string(requested)
                                     << ": a dimension of size 0 makes it ambiguous");
    KAEL_CHECK(element_count % known_count == 0, "shape " << to_string(requested)
                                                          << " does not fit " << element_count
                                                          << " elements");

    Shape result = requested;
    result[static_cast<std::size_t>(std::distance(requested.begin(), inferred))] =
        element_count / known_count;
    return result;
}

} // namespace kael
