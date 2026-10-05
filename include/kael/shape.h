#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace kael{

    using Shape   = std::vector<std::int64_t>; // define std:: in case some compilers
    using Strides = std::vector<std::int64_t>; // can't find it

    // returns shape as string -> [1,2,3] : "[1, 2, 3]"
    std::string to_string(const Shape& shape);

}