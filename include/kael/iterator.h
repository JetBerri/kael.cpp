#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael {

struct LoopDims {
    Shape shape;
    std::vector<Strides> operand_strides;
};

[[nodiscard]] LoopDims coalesce_dims(const LoopDims& dims);

class TensorIterator {
public:
    TensorIterator& add_output(Tensor output);
    TensorIterator& allocate_output(DType dtype);
    TensorIterator& add_input(Tensor input);
    TensorIterator& build();

    [[nodiscard]] const Tensor& output() const;
    [[nodiscard]] const LoopDims& dims() const noexcept { return dims_; }

    template <typename InnerLoop> void run(InnerLoop&& inner_loop) const;

private:
    void check_inputs() const;
    [[nodiscard]] Shape common_input_shape() const;
    void prepare_output(const Shape& common_shape);
    void prepare_loop(const Shape& common_shape);

    std::optional<Tensor> output_;
    std::optional<DType> output_dtype_;
    std::vector<Tensor> inputs_;
    std::vector<std::byte*> base_pointers_;
    LoopDims dims_;
    std::int64_t element_count_ = 0;
};

template <typename InnerLoop> void TensorIterator::run(InnerLoop&& inner_loop) const {
    if (element_count_ == 0) {
        return;
    }

    const std::size_t operand_count = base_pointers_.size();
    const std::size_t outer_dims = dims_.shape.size() - 1;
    const std::int64_t inner_size = dims_.shape[outer_dims];

    std::vector<std::int64_t> inner_strides(operand_count);
    for (std::size_t op = 0; op < operand_count; ++op) {
        inner_strides[op] = dims_.operand_strides[op][outer_dims];
    }

    std::vector<std::int64_t> counter(outer_dims, 0);
    std::vector<std::int64_t> byte_offsets(operand_count, 0);
    std::vector<std::byte*> pointers(operand_count);

    for (std::int64_t step = 0; step < element_count_ / inner_size; ++step) {
        for (std::size_t op = 0; op < operand_count; ++op) {
            pointers[op] = base_pointers_[op] + byte_offsets[op];
        }
        inner_loop(pointers.data(), inner_strides.data(), inner_size);

        for (std::size_t dim = outer_dims; dim-- > 0;) {
            ++counter[dim];
            for (std::size_t op = 0; op < operand_count; ++op) {
                byte_offsets[op] += dims_.operand_strides[op][dim];
            }
            if (counter[dim] < dims_.shape[dim]) {
                break;
            }
            for (std::size_t op = 0; op < operand_count; ++op) {
                byte_offsets[op] -= dims_.operand_strides[op][dim] * dims_.shape[dim];
            }
            counter[dim] = 0;
        }
    }
}

} // namespace kael
