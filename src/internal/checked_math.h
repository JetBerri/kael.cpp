#pragma once

#include <cstdint>
#include <limits>

#include "kael/error.h"

namespace kael::detail {

inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const char* quantity) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_mul expects non-negative operands for " << quantity);
    if (a == 0 || b == 0) {
        return 0;
    }
    KAEL_CHECK(a <= std::numeric_limits<std::int64_t>::max() / b,
               quantity << " overflows int64 (" << a << " * " << b << ")");
    return a * b;
}

inline std::int64_t checked_add(std::int64_t a, std::int64_t b, const char* quantity) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_add expects non-negative operands for " << quantity);
    KAEL_CHECK(a <= std::numeric_limits<std::int64_t>::max() - b,
               quantity << " overflows int64 (" << a << " + " << b << ")");
    return a + b;
}

} // namespace kael::detail
