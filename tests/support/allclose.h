#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <type_traits>

#include <gtest/gtest.h>

#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael::test {

struct Tolerance {
    double relative;
    double absolute;
};

inline Tolerance default_tolerance(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return {1e-5, 1e-6};
    case DType::F64:
        return {1e-12, 1e-14};
    case DType::I64:
    case DType::Bool:
        return {0.0, 0.0};
    }
    detail::unreachable();
}

struct AllcloseOptions {
    bool equal_nan = false;
    std::optional<Tolerance> tolerance = std::nullopt;
};

template <typename T>
bool values_close(T actual, T expected, const Tolerance& tolerance, bool equal_nan) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(actual) || std::isnan(expected)) {
            return equal_nan && std::isnan(actual) && std::isnan(expected);
        }
        if (std::isinf(actual) || std::isinf(expected)) {
            return actual == expected;
        }
        const double difference =
            std::abs(static_cast<double>(actual) - static_cast<double>(expected));
        return difference <=
               tolerance.absolute + tolerance.relative * std::abs(static_cast<double>(expected));
    } else {
        return actual == expected;
    }
}

inline testing::AssertionResult allclose(const Tensor& actual, const Tensor& expected,
                                         AllcloseOptions options = {}) {
    if (actual.shape() != expected.shape()) {
        return testing::AssertionFailure()
               << "shape " << to_string(actual.shape()) << " != " << to_string(expected.shape());
    }
    if (actual.dtype() != expected.dtype()) {
        return testing::AssertionFailure()
               << "dtype " << actual.dtype() << " != " << expected.dtype();
    }

    const Tolerance tolerance = options.tolerance.value_or(default_tolerance(actual.dtype()));
    return dispatch(actual.dtype(), [&]<typename T>() -> testing::AssertionResult {
        for (std::int64_t linear = 0; linear < actual.numel(); ++linear) {
            const Shape index = index_from_linear(linear, actual.shape());
            const T a = actual.at<T>(index);
            const T e = expected.at<T>(index);
            if (!values_close<T>(a, e, tolerance, options.equal_nan)) {
                return testing::AssertionFailure()
                       << "element " << to_string(index) << ": " << +a << " != " << +e
                       << " (difference " << +(static_cast<double>(a) - static_cast<double>(e))
                       << ")";
            }
        }
        return testing::AssertionSuccess();
    });
}

} // namespace kael::test
