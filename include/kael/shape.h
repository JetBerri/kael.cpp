#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace kael {

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

std::string to_string(const Shape& shape);
void validate_shape(const Shape& shape);
std::int64_t numel(const Shape& shape);
Strides contiguous_strides(const Shape& shape);

std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                               std::span<const std::int64_t> strides, std::int64_t base_offset,
                               std::span<const std::int64_t> index);
Shape index_from_linear(std::int64_t linear, const Shape& shape);

Shape broadcast_shapes(const Shape& a, const Shape& b);
Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target);

std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim);
Shape infer_shape(const Shape& requested, std::int64_t element_count);

} // namespace kael
