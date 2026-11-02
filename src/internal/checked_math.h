#pragma once

#include <cstdint>
#include <limits>

#include "kael/error.h"

// Overflow-checked int64 arithmetic for size and offset computations.
// Internal header: not installed, not part of the public API.
namespace kael::detail {

// a * b for non-negative a and b; throws if the product overflows int64.
// `what` names the quantity in the error message.
inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const char* what) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_mul expects non-negative operands for " << what);
    if (a == 0 || b == 0) {
        return 0;
    }
    KAEL_CHECK(a <= std::numeric_limits<std::int64_t>::max() / b,
               what << " overflows int64 (" << a << " * " << b << ")");
    return a * b;
}

// a + b for non-negative a and b; throws if the sum overflows int64.
inline std::int64_t checked_add(std::int64_t a, std::int64_t b, const char* what) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_add expects non-negative operands for " << what);
    KAEL_CHECK(a <= std::numeric_limits<std::int64_t>::max() - b,
               what << " overflows int64 (" << a << " + " << b << ")");
    return a + b;
}

} // namespace kael::detail
