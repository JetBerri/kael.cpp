#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "internal/arithmetic.h"
#include "internal/copy.h"
#include "internal/index_checks.h"
#include "internal/operands.h"

namespace kael {

namespace {

int category(DType dtype) {
    if (dtype == DType::Bool) {
        return 0;
    }
    return dtype == DType::I64 ? 1 : 2;
}

void check_writable(const Tensor& self, const char* op) {
    KAEL_CHECK(!self.has_internal_overlap(),
               op << " cannot write to shape " << to_string(self.shape()) << " with strides "
                  << to_string(self.strides())
                  << " because several positions share one element; call clone() first");
}

void check_result_fits(DType result, DType self, const char* op) {
    KAEL_CHECK(category(result) <= category(self), op << " result of dtype " << result
                                                      << " cannot be stored in a tensor of dtype "
                                                      << self);
}

bool same_layout(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() && a.strides() == b.strides() && a.offset() == b.offset();
}

Tensor without_partial_aliasing(const Tensor& self, const Tensor& input) {
    return input.shares_storage(self) && !same_layout(input, self) ? input.clone() : input;
}

void check_scatter_source(const Tensor& indices, const Tensor& source) {
    KAEL_CHECK(source.dim() == indices.dim(),
               "scatter_add_ source shape " << to_string(source.shape()) << " and indices shape "
                                            << to_string(indices.shape())
                                            << " have different number of dimensions");
    for (std::size_t i = 0; i < indices.shape().size(); ++i) {
        KAEL_CHECK(indices.shape()[i] <= source.shape()[i], "scatter_add_ indices shape "
                                                                << to_string(indices.shape())
                                                                << " is larger than source shape "
                                                                << to_string(source.shape()));
    }
}

template <typename F>
Tensor& binary_inplace(Tensor& self, const Tensor& other, DType result, const char* op, F f) {
    check_writable(self, op);
    check_result_fits(result, self.dtype(), op);
    binary_kernel_out(self, self, without_partial_aliasing(self, other), f);
    return self;
}

} // namespace

Tensor& Tensor::add_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, detail::add_dtype(self.dtype(), rhs.dtype()), "add_",
                          detail::kPlus);
}

Tensor& Tensor::sub_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    detail::check_no_bool(self, rhs, "sub_");
    return binary_inplace(*this, rhs, promote_types(self.dtype(), rhs.dtype()), "sub_",
                          detail::kMinus);
}

Tensor& Tensor::mul_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, promote_types(self.dtype(), rhs.dtype()), "mul_",
                          detail::kTimes);
}

Tensor& Tensor::div_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, detail::div_dtype(self.dtype(), rhs.dtype()), "div_",
                          detail::kDivides);
}

Tensor& Tensor::pow_(const Operand& exponent) {
    const auto [self, rhs] = detail::resolve_operands(*this, exponent);
    detail::check_no_bool(self, rhs, "pow_");
    return binary_inplace(*this, rhs, promote_types(self.dtype(), rhs.dtype()), "pow_",
                          detail::kPower);
}

Tensor& Tensor::clamp_(Scalar min, Scalar max) {
    check_writable(*this, "clamp_");
    check_result_fits(promote_with_scalar(promote_with_scalar(dtype(), min), max), dtype(),
                      "clamp_");
    unary_kernel_out(*this, *this, detail::Clamper{min.value(), max.value()});
    return *this;
}

Tensor& Tensor::copy_(const Tensor& source) {
    check_writable(*this, "copy_");
    detail::convert_strided(*this, without_partial_aliasing(*this, source).expand(shape()));
    return *this;
}

Tensor& Tensor::fill_(Scalar value) {
    return copy_(scalar(value.value(), dtype()));
}

Tensor& Tensor::zero_() {
    return fill_(0);
}

Tensor& Tensor::masked_fill_(const Tensor& mask, Scalar value) {
    KAEL_CHECK(mask.dtype() == DType::Bool, "masked_fill needs a bool mask, got " << mask.dtype());
    check_writable(*this, "masked_fill_");
    binary_kernel_out(*this, *this, mask,
                      [fill = value.value()]<typename T>(T current, T selected) -> T {
                          return selected != T{0} ? convert_value<T>(fill) : current;
                      });
    return *this;
}

Tensor& Tensor::scatter_add_(std::int64_t dim, const Tensor& indices, const Tensor& source) {
    detail::check_index_tensor(indices, "scatter_add_");
    const std::size_t position = dim_position(dim, this->dim());
    detail::check_gather_shapes(*this, indices, position, "scatter_add_");
    check_scatter_source(indices, source);

    const Tensor converted = source.to(dtype());
    dispatch(dtype(), [&]<typename T>() {
        detail::for_each_index(indices.shape(), [&](const Shape& index) {
            Shape target = index;
            target[position] = detail::checked_index(indices.at<std::int64_t>(index),
                                                     shape()[position], "scatter_add_");
            T& element = at<T>(target);
            element = static_cast<T>(element + converted.at<T>(index));
        });
    });
    return *this;
}

Tensor& operator+=(Tensor& t, const Operand& other) {
    return t.add_(other);
}

Tensor& operator-=(Tensor& t, const Operand& other) {
    return t.sub_(other);
}

Tensor& operator*=(Tensor& t, const Operand& other) {
    return t.mul_(other);
}

Tensor& operator/=(Tensor& t, const Operand& other) {
    return t.div_(other);
}

} // namespace kael
