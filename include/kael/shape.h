#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <string>

namespace kael{

    using Shape   = std::vector<std::int64_t>; // define std:: in case some compilers
    using Strides = std::vector<std::int64_t>; // can't find it

    // returns shape as text, e.g. [1,2,3] : "[1, 2, 3]"
    std::string to_string(const Shape& shape);

    // catches shape and throws error if negative dimension
    void validate_shape(const Shape& shape);

    // elements in shape and overflow check
    std::int64_t numel(const Shape& shape);

    Strides contiguous_strides(const Shape& shape);

    // position of a multidimensional index in the flat array:
    // base_offset + sum(index[i] * strides[i])
    std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                                   std::span<const std::int64_t> strides,
                                   std::int64_t base_offset,
                                   std::span<const std::int64_t> index);

    // multidimensional index of the element at position `linear` in
    // row-major order, e.g. linear 5 in shape [2, 3] : [1, 2];
    // inverse of offset_from_index with contiguous strides and base 0;
    // throws if the shape is invalid or linear is outside [0, numel(shape))
    Shape index_from_linear(std::int64_t linear,
                            const Shape& shape);

    // shape resulting from broadcasting a and b, aligned from the right
    // (missing leading dimensions count as 1); throws if a dimension is
    // negative or if a and b are not broadcastable
    Shape broadcast_shapes(const Shape& a, 
                           const Shape& b);

    Strides broadcast_strides(const Shape& shape,
                              const Strides& strides,
                              const Shape& target);

}