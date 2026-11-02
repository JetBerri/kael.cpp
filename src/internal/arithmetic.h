#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include "kael/dtype.h"
#include "kael/error.h"
#include "kael/kernels.h"
#include "kael/operand.h"
#include "kael/tensor.h"

#include "internal/element_functions.h"

namespace kael::detail {

inline constexpr functions::Plus kPlus;
inline constexpr functions::Minus kMinus;
inline constexpr functions::Times kTimes;
inline constexpr functions::Divides kDivides;
inline constexpr functions::Power kPower;

inline void clamp_into(Tensor& out, const Tensor& input, Scalar min, Scalar max) {
    TensorIterator iterator;
    iterator.add_output(out).add_input(input).build();
    dispatch(out.dtype(), [&]<typename T>() {
        functions::Clamp<T> clamp{min.as<T>(), max.as<T>()};
        run_typed<T, T, 1>(iterator, clamp);
    });
}

inline void check_no_bool(const Tensor& a, const Tensor& b, const char* op) {
    KAEL_CHECK_TYPE(a.dtype() != DType::Bool && b.dtype() != DType::Bool,
                    op << " does not support dtype bool");
}

inline void check_integer_exponent(const Tensor& base, const Operand& exponent) {
    if (exponent.is_tensor() || is_floating_point(base.dtype())) {
        return;
    }
    KAEL_CHECK_VALUE(is_floating_point(exponent.scalar().dtype()) ||
                         exponent.scalar().as<std::int64_t>() >= 0,
                     "integers to negative integer powers are not allowed");
}

inline DType add_dtype(const Tensor& a, const Tensor& b) {
    return a.dtype() == DType::Bool && b.dtype() == DType::Bool ? DType::I64
                                                                : result_dtype({&a, &b});
}

inline DType div_dtype(const Tensor& a, const Tensor& b) {
    const DType promoted = result_dtype({&a, &b});
    return is_floating_point(promoted) ? promoted : DType::F32;
}

} // namespace kael::detail
