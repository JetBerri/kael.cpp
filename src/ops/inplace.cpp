#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "internal/arithmetic.h"
#include "internal/copy.h"
#include "internal/index_checks.h"
#include "internal/operands.h"

namespace kael {

namespace {

void check_grad_rules(const Tensor& self, const Tensor* source, const char* op) {
    if (!autograd::GradMode::is_enabled()) {
        return;
    }
    KAEL_CHECK(!self.requires_grad() || !self.is_leaf(),
               "a leaf tensor that requires grad is being used in the in-place operation " << op);
    KAEL_CHECK(
        !self.requires_grad(),
        op << " is not supported on a tensor with gradient history; use the out-of-place op");
    KAEL_CHECK(source == nullptr || !source->requires_grad(),
               op << " is not supported with a source that requires grad");
}

void check_writable(const Tensor& self, const char* op) {
    KAEL_CHECK_VALUE(!self.has_internal_overlap(),
                     op << " cannot write to shape " << to_string(self.shape()) << " with strides "
                        << to_string(self.strides())
                        << " because several positions share one element; call clone() first");
}

void check_result_fits(DType result, DType self, const char* op) {
    KAEL_CHECK_TYPE(dtype_category(result) <= dtype_category(self),
                    op << " result of dtype " << result << " cannot be stored in a tensor of dtype "
                       << self);
}

bool same_layout(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() && a.strides() == b.strides() && a.offset() == b.offset();
}

Tensor without_partial_aliasing(const Tensor& self, const Tensor& input) {
    return input.shares_storage(self) && !same_layout(input, self) ? input.clone() : input;
}

void check_scatter_source(const Tensor& indices, const Tensor& source) {
    KAEL_CHECK_VALUE(source.dim() == indices.dim(),
                     "scatter_add_ source shape "
                         << to_string(source.shape()) << " and indices shape "
                         << to_string(indices.shape()) << " have different number of dimensions");
    for (std::size_t i = 0; i < indices.shape().size(); ++i) {
        KAEL_CHECK_VALUE(indices.shape()[i] <= source.shape()[i],
                         "scatter_add_ indices shape " << to_string(indices.shape())
                                                       << " is larger than source shape "
                                                       << to_string(source.shape()));
    }
}

template <typename F>
Tensor& binary_inplace(Tensor& self, const Tensor& other, DType result, const char* op, F f) {
    check_grad_rules(self, &other, op);
    check_writable(self, op);
    check_result_fits(result, self.dtype(), op);
    binary_kernel_out(self, self, without_partial_aliasing(self, other), f);
    self.impl()->storage->bump_version();
    return self;
}

} // namespace

Tensor& Tensor::add_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, detail::add_dtype(self, rhs), "add_", detail::kPlus);
}

Tensor& Tensor::sub_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    detail::check_no_bool(self, rhs, "sub_");
    return binary_inplace(*this, rhs, result_dtype({&self, &rhs}), "sub_", detail::kMinus);
}

Tensor& Tensor::mul_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, result_dtype({&self, &rhs}), "mul_", detail::kTimes);
}

Tensor& Tensor::div_(const Operand& other) {
    const auto [self, rhs] = detail::resolve_operands(*this, other);
    return binary_inplace(*this, rhs, detail::div_dtype(self, rhs), "div_", detail::kDivides);
}

Tensor& Tensor::pow_(const Operand& exponent) {
    const auto [self, rhs] = detail::resolve_operands(*this, exponent);
    detail::check_no_bool(self, rhs, "pow_");
    detail::check_integer_exponent(self, exponent);
    return binary_inplace(*this, rhs, result_dtype({&self, &rhs}), "pow_", detail::kPower);
}

Tensor& Tensor::clamp_(Scalar min, Scalar max) {
    check_grad_rules(*this, nullptr, "clamp_");
    check_writable(*this, "clamp_");
    check_result_fits(promote_with_scalar(promote_with_scalar(dtype(), min), max), dtype(),
                      "clamp_");
    detail::clamp_into(*this, *this, min, max);
    impl_->storage->bump_version();
    return *this;
}

Tensor& Tensor::copy_(const Tensor& source) {
    check_grad_rules(*this, &source, "copy_");
    check_writable(*this, "copy_");
    const autograd::NoGradGuard guard;
    const Tensor same_device = source.to(device());
    detail::convert_strided(*this, without_partial_aliasing(*this, same_device).expand(shape()));
    impl_->storage->bump_version();
    return *this;
}

Tensor& Tensor::fill_(Scalar value) {
    return copy_(scalar(value, dtype()));
}

Tensor& Tensor::zero_() {
    return fill_(0);
}

Tensor& Tensor::masked_fill_(const Tensor& mask, Scalar value) {
    KAEL_CHECK_TYPE(mask.dtype() == DType::Bool,
                    "masked_fill needs a bool mask, got " << mask.dtype());
    check_grad_rules(*this, nullptr, "masked_fill_");
    check_writable(*this, "masked_fill_");

    TensorIterator iterator;
    iterator.add_output(*this)
        .add_input(*this)
        .add_input(without_partial_aliasing(*this, mask).to(dtype()))
        .build();
    dispatch(dtype(), [&]<typename T>() {
        functions::MaskedFill<T> fill{value.as<T>()};
        detail::run_typed<T, T, 2>(iterator, fill);
    });
    impl_->storage->bump_version();
    return *this;
}

Tensor& Tensor::scatter_add_(std::int64_t dim, const Tensor& indices, const Tensor& source) {
    detail::check_index_tensor(indices, "scatter_add_");
    check_grad_rules(*this, &source, "scatter_add_");
    const std::size_t position = dim_position(dim, this->dim());
    detail::check_gather_shapes(*this, indices, position, "scatter_add_");
    check_scatter_source(indices, source);
    check_writable(*this, "scatter_add_");
    KAEL_CHECK_TYPE(source.dtype() == dtype(), "scatter_add_ needs a source of dtype "
                                                   << dtype() << ", got " << source.dtype());
    KAEL_CHECK(!source.shares_storage(*this) && !indices.shares_storage(*this),
               "scatter_add_ cannot read from the tensor it writes to; call clone() first");
    check_same_device(*this, indices);
    check_same_device(*this, source);

    if (device() != Device::CPU) {
        if (dtype() == DType::Bool) {
            Tensor host = cpu();
            host.scatter_add_(dim, indices.cpu(), source.cpu());
            const autograd::NoGradGuard guard;
            return copy_(host);
        }
        const autograd::NoGradGuard guard;
        const Tensor contiguous_indices = indices.contiguous();
        detail::check_device_indices(contiguous_indices, shape()[position], "scatter_add_");
        device::scatter_add(dtype(), device_data(),
                            reinterpret_cast<const std::int64_t*>(contiguous_indices.device_data()),
                            source.device_data(),
                            detail::index_layout(*this, contiguous_indices, source, position));
        impl_->storage->bump_version();
        return *this;
    }

    dispatch(dtype(), [&]<typename T>() {
        detail::for_each_index(indices.shape(), [&](const Shape& index) {
            Shape target = index;
            target[position] = detail::checked_index(indices.at<std::int64_t>(index),
                                                     shape()[position], "scatter_add_");
            T& element = at<T>(target);
            element = convert_value<T>(element + source.at<T>(index));
        });
    });
    impl_->storage->bump_version();
    return *this;
}

Tensor& Tensor::index_add_(std::int64_t dim, const Tensor& indices, const Tensor& source) {
    detail::check_index_tensor(indices, "index_add_");
    KAEL_CHECK_VALUE(indices.dim() == 1,
                     "index_add_ needs 1-d indices, got shape " << to_string(indices.shape()));
    const std::size_t position = dim_position(dim, this->dim());
    KAEL_CHECK_VALUE(source.dim() == this->dim() && source.shape()[position] == indices.numel(),
                     "index_add_ source shape " << to_string(source.shape()) << " does not match "
                                                << indices.numel() << " indices at dimension "
                                                << dim);
    Shape along_dim(shape().size(), 1);
    along_dim[position] = indices.numel();
    const autograd::NoGradGuard guard;
    return scatter_add_(dim, indices.view(along_dim).expand(source.shape()), source);
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
