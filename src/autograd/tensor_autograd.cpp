#include <memory>

#include "kael/autograd/engine.h"
#include "kael/autograd/node.h"
#include "kael/tensor.h"

namespace kael {

namespace {

autograd::AutogradMeta& meta_of(TensorImpl& impl) {
    if (!impl.autograd) {
        impl.autograd = std::make_unique<autograd::AutogradMeta>();
    }
    return *impl.autograd;
}

} // namespace

bool Tensor::requires_grad() const noexcept {
    return impl_->autograd && impl_->autograd->requires_grad;
}

Tensor& Tensor::requires_grad_(bool required) {
    KAEL_CHECK(is_leaf() || required == requires_grad(),
               "you can only change requires_grad flags of leaf tensors");
    KAEL_CHECK(!required || is_floating_point(dtype()),
               "only floating point tensors can require gradients, got dtype " << dtype());
    if (required || impl_->autograd) {
        meta_of(*impl_).requires_grad = required;
    }
    return *this;
}

bool Tensor::is_leaf() const noexcept {
    return grad_fn() == nullptr;
}

std::optional<Tensor> Tensor::grad() const {
    return impl_->autograd ? impl_->autograd->grad : std::nullopt;
}

std::shared_ptr<autograd::Node> Tensor::grad_fn() const noexcept {
    return impl_->autograd ? impl_->autograd->grad_fn : nullptr;
}

void Tensor::zero_grad() {
    if (impl_->autograd) {
        impl_->autograd->grad.reset();
    }
}

void Tensor::retain_grad() {
    KAEL_CHECK(requires_grad(), "retain_grad needs a tensor that requires grad");
    if (const auto function = grad_fn()) {
        function->retain_output(impl_);
    }
}

void Tensor::backward(bool retain_graph) const {
    autograd::backward(*this, std::nullopt, retain_graph);
}

void Tensor::backward(const Tensor& gradient, bool retain_graph) const {
    autograd::backward(*this, gradient, retain_graph);
}

Tensor& Tensor::detach_() {
    impl_->autograd.reset();
    return *this;
}

std::int64_t Tensor::version() const noexcept {
    return impl_->storage->version();
}

} // namespace kael
