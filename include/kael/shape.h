#pragma once

#include <vector>
#include <cstdint>
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

}