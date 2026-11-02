#include "internal/copy.h"

namespace kael::detail {

namespace {

// Slow but obviously correct: computes both offsets from scratch for every
// element. Phase 5 replaces this with the iterator.
template <typename To, typename From>
void convert_loop(Tensor& dst, const Tensor& src) {
    const Storage& src_storage = *src.impl()->storage;
    const From* src_base = reinterpret_cast<const From*>(src_storage.data());
    To* dst_base = reinterpret_cast<To*>(dst.impl()->storage->data());

    const std::int64_t n = src.numel();
    for (std::int64_t linear = 0; linear < n; ++linear) {
        const Shape index = index_from_linear(linear, src.shape());
        const std::int64_t from =
            offset_from_index(src.shape(), src.strides(), src.offset(), index);
        const std::int64_t to = offset_from_index(dst.shape(), dst.strides(), dst.offset(), index);
        dst_base[to] = convert_value<To>(src_base[from]);
    }
}

} // namespace

void copy_strided(Tensor& dst, const Tensor& src) {
    KAEL_CHECK(dst.dtype() == src.dtype(),
               "copy_strided expects the same dtype, got " << dst.dtype() << " and " << src.dtype());
    convert_strided(dst, src);
}

void convert_strided(Tensor& dst, const Tensor& src) {
    KAEL_CHECK(dst.shape() == src.shape(), "copy expects the same shape, got "
                                               << to_string(dst.shape()) << " and "
                                               << to_string(src.shape()));
    dispatch(src.dtype(), [&]<typename From>() {
        dispatch(dst.dtype(), [&]<typename To>() { convert_loop<To, From>(dst, src); });
    });
}

} // namespace kael::detail
