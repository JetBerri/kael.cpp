#include "kael/tensor.h"

#include <cstring>
#include <utility>
#include <vector>

#include "kael/autograd/grad_mode.h"
#include "kael/device_kernels.h"
#include "kael/factory.h"

#include "autograd/record.h"
#include "internal/copy.h"

namespace kael {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {
    KAEL_CHECK(impl_ != nullptr, "cannot create a tensor from a null impl");
    impl_->validate();
}

Tensor Tensor::from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(std::move(impl));
}

std::int64_t Tensor::size(std::int64_t dim) const {
    return shape()[dim_position(dim, this->dim())];
}

std::int64_t Tensor::stride(std::int64_t dim) const {
    return strides()[dim_position(dim, this->dim())];
}

std::int64_t Tensor::nbytes() const {
    return numel() * element_size(dtype());
}

std::byte* Tensor::first_element() const {
    std::byte* base = impl_->storage->data();
    // An empty view can have an offset past the end of its storage, or a null storage.
    return numel() == 0 ? base : base + offset() * element_size(dtype());
}

void Tensor::check_single_element() const {
    KAEL_CHECK_VALUE(numel() == 1, "item needs a tensor with one element, got shape "
                                       << to_string(shape()) << " with " << numel() << " elements");
}

void Tensor::check_on_host() const {
    KAEL_CHECK(device() == Device::CPU, "this operation reads tensor memory directly, which is not "
                                        "possible on device "
                                            << device() << "; call cpu() first");
}

double Tensor::item() const {
    check_single_element();
    if (device() != Device::CPU) {
        return cpu().item();
    }
    return dispatch(dtype(), [&]<typename T>() {
        return static_cast<double>(*reinterpret_cast<const T*>(first_element()));
    });
}

std::int64_t Tensor::element_offset(std::span<const std::int64_t> index) const {
    KAEL_CHECK_VALUE(static_cast<std::int64_t>(index.size()) == dim(),
                     "got " << index.size() << " indices for a tensor with " << dim()
                            << " dimensions");
    std::vector<std::int64_t> wrapped(index.begin(), index.end());
    for (std::size_t i = 0; i < wrapped.size(); ++i) {
        if (wrapped[i] < 0) {
            wrapped[i] += shape()[i];
        }
    }
    return offset_from_index(shape(), strides(), offset(), wrapped);
}

Tensor Tensor::to(DType target) const {
    if (target == dtype()) {
        return *this;
    }
    Tensor out = empty(shape(), target, device());
    detail::convert_strided(out, *this);
    autograd::record_to(out, *this);
    return out;
}

Tensor Tensor::to(Device target) const {
    if (target == device()) {
        return *this;
    }
    const Tensor source = autograd::without_grad([&] { return detach().contiguous(); });
    Tensor out = empty(shape(), dtype(), target);
    if (nbytes() > 0) {
        if (target == Device::CPU) {
            device::copy_to_host(out.device_data(), source.device_data(), nbytes());
        } else {
            device::copy_to_device(out.device_data(), source.device_data(), nbytes());
        }
    }
    autograd::record_to_device(out, *this);
    return out;
}

Tensor& Tensor::move_to_(Device target) {
    if (target == device()) {
        return *this;
    }
    const Tensor moved = detach().to(target);
    impl_->storage = moved.impl()->storage;
    impl_->strides = moved.strides();
    impl_->offset = moved.offset();
    impl_->device = target;
    return *this;
}

Tensor Tensor::clone() const {
    Tensor out = empty(shape(), dtype(), device());
    if (!is_contiguous()) {
        detail::convert_strided(out, *this);
    } else if (nbytes() > 0) {
        if (device() == Device::CPU) {
            std::memcpy(out.data_ptr_raw(), data_ptr_raw(), static_cast<std::size_t>(nbytes()));
        } else {
            device::copy_within_device(out.device_data(), device_data(), nbytes());
        }
    }
    autograd::record_clone(out, *this);
    return out;
}

Tensor Tensor::contiguous() const {
    return is_contiguous() ? *this : clone();
}

} // namespace kael
