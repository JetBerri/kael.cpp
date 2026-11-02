#include "kael/iterator.h"

#include <algorithm>
#include <utility>

#include "kael/factory.h"

namespace kael {

namespace {

LoopDims without_size_one_dims(const LoopDims& dims) {
    LoopDims result;
    result.operand_strides.resize(dims.operand_strides.size());
    for (std::size_t dim = 0; dim < dims.shape.size(); ++dim) {
        if (dims.shape[dim] == 1) {
            continue;
        }
        result.shape.push_back(dims.shape[dim]);
        for (std::size_t op = 0; op < dims.operand_strides.size(); ++op) {
            result.operand_strides[op].push_back(dims.operand_strides[op][dim]);
        }
    }
    return result;
}

bool can_merge_with_next(const LoopDims& dims, std::size_t outer) {
    return std::ranges::all_of(dims.operand_strides, [&](const Strides& strides) {
        return strides[outer] == strides[outer + 1] * dims.shape[outer + 1];
    });
}

Strides to_bytes(Strides strides, DType dtype) {
    for (std::int64_t& stride : strides) {
        stride *= element_size(dtype);
    }
    return strides;
}

} // namespace

void check_same_device(const Tensor& a, const Tensor& b) {
    KAEL_CHECK(a.device() == b.device(), "expected all tensors to be on the same device, but found "
                                             << a.device() << " and " << b.device());
}

LoopDims coalesce_dims(const LoopDims& dims) {
    LoopDims merged = without_size_one_dims(dims);

    for (std::size_t inner = merged.shape.size(); inner-- > 1;) {
        const std::size_t outer = inner - 1;
        if (!can_merge_with_next(merged, outer)) {
            continue;
        }
        merged.shape[outer] *= merged.shape[inner];
        merged.shape.erase(merged.shape.begin() + static_cast<std::ptrdiff_t>(inner));
        for (Strides& strides : merged.operand_strides) {
            strides[outer] = strides[inner];
            strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(inner));
        }
    }
    return merged;
}

TensorIterator& TensorIterator::add_output(Tensor output) {
    output_ = std::move(output);
    return *this;
}

TensorIterator& TensorIterator::allocate_output(DType dtype) {
    output_dtype_ = dtype;
    return *this;
}

TensorIterator& TensorIterator::add_input(Tensor input) {
    inputs_.push_back(std::move(input));
    return *this;
}

TensorIterator& TensorIterator::build() {
    check_inputs();
    const Shape common_shape = common_input_shape();
    prepare_output(common_shape);
    prepare_loop(common_shape);
    return *this;
}

const Tensor& TensorIterator::output() const {
    KAEL_CHECK(output_.has_value(), "iterator output is not available before build");
    return *output_;
}

void TensorIterator::check_inputs() const {
    KAEL_CHECK(!inputs_.empty(), "iterator needs at least one input");
    KAEL_CHECK(output_.has_value() || output_dtype_.has_value(),
               "iterator needs an output or an output dtype");
    for (const Tensor& input : inputs_) {
        KAEL_CHECK_TYPE(input.dtype() == inputs_.front().dtype(),
                        "iterator inputs must share a dtype, got " << inputs_.front().dtype()
                                                                   << " and " << input.dtype());
        check_same_device(input, inputs_.front());
    }
    if (output_.has_value()) {
        check_same_device(*output_, inputs_.front());
    }
}

void TensorIterator::check_runs_on_host() const {
    KAEL_CHECK(!on_device(), "this kernel has no implementation for device " << output().device());
}

device::ElementLayout TensorIterator::device_layout() const {
    KAEL_CHECK(static_cast<int>(dims_.shape.size()) <= device::kMaxDims,
               "device kernels support at most " << device::kMaxDims
                                                 << " dimensions after merging");
    KAEL_CHECK(static_cast<int>(base_pointers_.size()) <= device::kMaxOperands,
               "device kernels support at most " << device::kMaxOperands << " operands");
    device::ElementLayout layout{};
    layout.dims = static_cast<int>(dims_.shape.size());
    layout.operands = static_cast<int>(base_pointers_.size());
    layout.count = element_count_;
    for (std::size_t dim = 0; dim < dims_.shape.size(); ++dim) {
        layout.shape[dim] = dims_.shape[dim];
    }
    for (std::size_t op = 0; op < base_pointers_.size(); ++op) {
        layout.base[op] = base_pointers_[op];
        for (std::size_t dim = 0; dim < dims_.shape.size(); ++dim) {
            layout.strides[op][dim] = dims_.operand_strides[op][dim];
        }
    }
    return layout;
}

Shape TensorIterator::common_input_shape() const {
    Shape common = inputs_.front().shape();
    for (const Tensor& input : inputs_) {
        common = broadcast_shapes(common, input.shape());
    }
    return common;
}

void TensorIterator::prepare_output(const Shape& common_shape) {
    if (!output_.has_value()) {
        output_ = empty(common_shape, *output_dtype_, inputs_.front().device());
        return;
    }
    KAEL_CHECK(output_->shape() == common_shape,
               "output shape " << to_string(output_->shape()) << " does not match the shape "
                               << to_string(common_shape) << " of the inputs");
}

void TensorIterator::prepare_loop(const Shape& common_shape) {
    base_pointers_.clear();
    LoopDims full;
    full.shape = common_shape;
    full.operand_strides.push_back(to_bytes(output_->strides(), output_->dtype()));
    base_pointers_.push_back(output_->device_data());
    for (Tensor& input : inputs_) {
        full.operand_strides.push_back(to_bytes(
            broadcast_strides(input.shape(), input.strides(), common_shape), input.dtype()));
        base_pointers_.push_back(input.device_data());
    }

    element_count_ = numel(common_shape);
    dims_ = coalesce_dims(full);
    if (dims_.shape.empty()) {
        dims_.shape.push_back(1);
        for (Strides& strides : dims_.operand_strides) {
            strides.push_back(0);
        }
    }
}

} // namespace kael
