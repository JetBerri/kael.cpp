#pragma once

#include <cstdint>
#include <vector>

#include "kael/shape.h"
#include "kael/tensor.h"

namespace kael::test {

template <typename T> std::vector<T> values(const Tensor& t) {
    std::vector<T> result;
    result.reserve(static_cast<std::size_t>(t.numel()));
    for (std::int64_t linear = 0; linear < t.numel(); ++linear) {
        result.push_back(t.at<T>(index_from_linear(linear, t.shape())));
    }
    return result;
}

} // namespace kael::test
