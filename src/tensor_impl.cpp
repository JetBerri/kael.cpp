#include "kael/tensor_impl.h"

#include "internal/checked_math.h"
#include "kael/error.h"

namespace kael {

bool TensorImpl::is_contiguous() const {
    if (numel() == 0) {
        return true; // no elements: nothing can be out of place
    }
    // From the innermost dimension outwards, each stride must equal the
    // number of elements of everything to its right. Size-1 dimensions are
    // skipped: you never step along them, so their stride does not matter.
    std::int64_t expected = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (shape[i] == 1) {
            continue;
        }
        if (strides[i] != expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

void TensorImpl::validate() const {
    KAEL_CHECK(storage != nullptr, "tensor has no storage");
    validate_shape(shape);
    KAEL_CHECK(shape.size() == strides.size(),
               "shape " << to_string(shape) << " and strides " << to_string(strides)
                        << " have different number of dimensions");
    for (std::size_t i = 0; i < strides.size(); ++i) {
        KAEL_CHECK(strides[i] >= 0, "negative stride " << strides[i] << " at dimension " << i
                                                       << " in strides " << to_string(strides)
                                                       << " (kael does not support them)");
    }
    KAEL_CHECK(offset >= 0, "negative storage offset " << offset);
    KAEL_CHECK(device == storage->device(), "tensor device " << device
                                                             << " does not match storage device "
                                                             << storage->device());

    if (numel() == 0) {
        return; // nothing is ever read: no bound to check
    }

    // Largest element offset reachable: offset + sum((shape[i] - 1) * strides[i]).
    std::int64_t max_offset = offset;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        max_offset = detail::checked_add(
            max_offset, detail::checked_mul(shape[i] - 1, strides[i], "tensor extent"),
            "tensor extent");
    }
    const std::int64_t needed =
        detail::checked_mul(max_offset + 1, element_size(dtype), "tensor extent in bytes");

    KAEL_CHECK(needed <= storage->nbytes(),
               "tensor with shape " << to_string(shape) << ", strides " << to_string(strides)
                                    << ", offset " << offset << " and dtype " << dtype
                                    << " needs " << needed << " bytes but storage has "
                                    << storage->nbytes());
}

} // namespace kael
