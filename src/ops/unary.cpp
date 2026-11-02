#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "autograd/record.h"
#include "internal/arithmetic.h"

namespace kael {

Tensor neg(const Tensor& t) {
    KAEL_CHECK_TYPE(t.dtype() != DType::Bool, "neg does not support dtype bool");
    Tensor out = unary_kernel(t, functions::Negate{});
    autograd::record_neg(out, t);
    return out;
}

Tensor abs(const Tensor& t) {
    Tensor out = unary_kernel(t, functions::Absolute{});
    autograd::record_abs(out, t);
    return out;
}

Tensor sign(const Tensor& t) {
    return unary_kernel(t, functions::Sign{});
}

Tensor exp(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Exp{});
    autograd::record_exp(out, t);
    return out;
}

Tensor log(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Log{});
    autograd::record_log(out, t);
    return out;
}

Tensor log1p(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Log1p{});
    autograd::record_log1p(out, t);
    return out;
}

Tensor expm1(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Expm1{});
    autograd::record_expm1(out, t);
    return out;
}

Tensor sqrt(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Sqrt{});
    autograd::record_sqrt(out, t);
    return out;
}

Tensor rsqrt(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Rsqrt{});
    autograd::record_rsqrt(out, t);
    return out;
}

Tensor sin(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Sin{});
    autograd::record_sin(out, t);
    return out;
}

Tensor cos(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Cos{});
    autograd::record_cos(out, t);
    return out;
}

Tensor tanh(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Tanh{});
    autograd::record_tanh(out, t);
    return out;
}

Tensor sigmoid(const Tensor& t) {
    Tensor out = floating_unary_kernel(t, functions::Sigmoid{});
    autograd::record_sigmoid(out, t);
    return out;
}

Tensor relu(const Tensor& t) {
    Tensor out = unary_kernel(t, functions::Relu{});
    autograd::record_relu(out, t);
    return out;
}

Tensor gelu(const Tensor& t, bool approximate) {
    Tensor out = approximate ? floating_unary_kernel(t, functions::GeluTanh{})
                             : floating_unary_kernel(t, functions::GeluExact{});
    autograd::record_gelu(out, t, approximate);
    return out;
}

Tensor clamp(const Tensor& t, Scalar min, Scalar max) {
    const DType dtype = promote_with_scalar(promote_with_scalar(t.dtype(), min), max);
    const Tensor input = t.to(dtype);
    Tensor out = empty_like(input);
    detail::clamp_into(out, input, min, max);
    autograd::record_clamp(out, t, min, max);
    return out;
}

Tensor operator-(const Tensor& t) {
    return neg(t);
}

} // namespace kael
