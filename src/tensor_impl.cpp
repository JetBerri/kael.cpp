#include "kael/tensor_impl.h"

#include "kael/error.h"

#include "internal/checked_math.h"

namespace kael {

namespace {

void check_fits_in_storage(const TensorImpl& impl) {
    std::int64_t last_element = impl.offset;
    for (std::size_t i = 0; i < impl.shape.size(); ++i) {
        const std::int64_t extent =
            detail::checked_mul(impl.shape[i] - 1, impl.strides[i], "tensor extent");
        last_element = detail::checked_add(last_element, extent, "tensor extent");
    }
    const std::int64_t bytes_needed =
        detail::checked_mul(last_element + 1, element_size(impl.dtype), "tensor extent in bytes");

    KAEL_CHECK(bytes_needed <= impl.storage->nbytes(),
               "tensor with shape " << to_string(impl.shape) << ", strides "
                                    << to_string(impl.strides) << ", offset " << impl.offset
                                    << " and dtype " << impl.dtype << " needs " << bytes_needed
                                    << " bytes but storage has " << impl.storage->nbytes());
}

} // namespace

bool TensorImpl::is_contiguous() const {
    if (numel() == 0) {
        return true;
    }
    std::int64_t expected_stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        // The stride of a size-1 dimension is never used to step, so any value is fine.
        if (shape[i] == 1) {
            continue;
        }
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

void TensorImpl::validate() const {
    KAEL_CHECK(storage != nullptr, "tensor has no storage");
    validate_shape(shape);
    KAEL_CHECK(shape.size() == strides.size(), "shape " << to_string(shape) << " and strides "
                                                        << to_string(strides)
                                                        << " have different number of dimensions");
    for (std::size_t i = 0; i < strides.size(); ++i) {
        KAEL_CHECK(strides[i] >= 0, "negative stride " << strides[i] << " at dimension " << i
                                                       << " in strides " << to_string(strides));
    }
    KAEL_CHECK(offset >= 0, "negative storage offset " << offset);
    KAEL_CHECK(device == storage->device(), "tensor device " << device
                                                             << " does not match storage device "
                                                             << storage->device());
    if (numel() > 0) {
        check_fits_in_storage(*this);
    }
}

} // namespace kael
