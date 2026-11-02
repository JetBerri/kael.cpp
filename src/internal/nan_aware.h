#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace kael::detail {

template <typename T> T nan_aware_max(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) ? a : b;
        }
    }
    return std::max(a, b);
}

template <typename T> T nan_aware_min(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) ? a : b;
        }
    }
    return std::min(a, b);
}

} // namespace kael::detail
