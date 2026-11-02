#include "internal/copy.h"

namespace kael::detail {

void copy_strided(Tensor& dst, const Tensor& src) {
    KAEL_CHECK(dst.dtype() == src.dtype(),
               "copy expects the same dtype, got " << dst.dtype() << " and " << src.dtype());
    convert_strided(dst, src);
}

void convert_strided(Tensor& dst, const Tensor& src) {
    KAEL_CHECK(dst.shape() == src.shape(), "copy expects the same shape, got "
                                               << to_string(dst.shape()) << " and "
                                               << to_string(src.shape()));
    dispatch(src.dtype(), [&]<typename From>() {
        dispatch(dst.dtype(), [&]<typename To>() {
            for (std::int64_t linear = 0; linear < src.numel(); ++linear) {
                const Shape index = index_from_linear(linear, src.shape());
                dst.at<To>(index) = convert_value<To>(src.at<From>(index));
            }
        });
    });
}

} // namespace kael::detail
