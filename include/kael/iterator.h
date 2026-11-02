#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "kael/dtype.h"
#include "kael/parallel.h"
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
    template <typename InnerLoop>
    void run_range(std::int64_t first, std::int64_t last, InnerLoop& inner_loop) const;

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
    parallel_for(0, element_count_, kElementGrain, [&](std::int64_t first, std::int64_t last) {
        run_range(first, last, inner_loop);
    });
}

template <typename InnerLoop>
void TensorIterator::run_range(std::int64_t first, std::int64_t last, InnerLoop& inner_loop) const {
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

    std::int64_t outer_position = first / inner_size;
    for (std::size_t dim = outer_dims; dim-- > 0;) {
        counter[dim] = outer_position % dims_.shape[dim];
        outer_position /= dims_.shape[dim];
        for (std::size_t op = 0; op < operand_count; ++op) {
            byte_offsets[op] += counter[dim] * dims_.operand_strides[op][dim];
        }
    }

    std::int64_t skipped_in_row = first % inner_size;
    for (std::int64_t remaining = last - first; remaining > 0;) {
        const std::int64_t count = std::min(inner_size - skipped_in_row, remaining);
        for (std::size_t op = 0; op < operand_count; ++op) {
            pointers[op] =
                base_pointers_[op] + byte_offsets[op] + skipped_in_row * inner_strides[op];
        }
        inner_loop(pointers.data(), inner_strides.data(), count);
        remaining -= count;
        skipped_in_row = 0;

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
