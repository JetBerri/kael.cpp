#include <algorithm>
#include <utility>

#include "internal/checked_math.h"
#include "kael/tensor.h"

namespace kael {

namespace {

// Every view ends here: a new impl over the same storage. Tensor::from_impl
// validates it, so a view with wrong metadata throws instead of reading
// someone else's memory.
Tensor make_view(const Tensor& base, Shape shape, Strides strides, std::int64_t offset) {
    auto impl = std::make_shared<TensorImpl>();
    impl->storage = base.impl()->storage;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->offset = offset;
    impl->dtype = base.dtype();
    impl->device = base.device();
    return Tensor::from_impl(std::move(impl));
}

} // namespace

Tensor Tensor::view(const Shape& requested) const {
    KAEL_CHECK(is_contiguous(), "view needs a contiguous tensor, but shape "
                                    << to_string(shape()) << " has strides "
                                    << to_string(strides()) << "; use reshape instead");
    Shape target = infer_shape(requested, numel());
    Strides target_strides = contiguous_strides(target);
    return make_view(*this, std::move(target), std::move(target_strides), offset());
}

Tensor Tensor::reshape(const Shape& requested) const {
    return contiguous().view(requested);
}

} // namespace kael
