#include "internal/copy.h"

#include "kael/kernels.h"

namespace kael::detail {

void copy_strided(Tensor& dst, const Tensor& src) {
    KAEL_CHECK(dst.dtype() == src.dtype(),
               "copy expects the same dtype, got " << dst.dtype() << " and " << src.dtype());
    convert_strided(dst, src);
}

void convert_strided(Tensor& dst, const Tensor& src) {
    TensorIterator iterator;
    iterator.add_output(dst).add_input(src).build();
    auto identity = [](auto value) { return value; };
    dispatch(src.dtype(), [&]<typename From>() {
        dispatch(dst.dtype(), [&]<typename To>() { run_typed<To, From, 1>(iterator, identity); });
    });
}

} // namespace kael::detail
