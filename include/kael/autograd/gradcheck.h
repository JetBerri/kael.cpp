#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "kael/tensor.h"

namespace kael::autograd {

struct GradcheckOptions {
    double epsilon = 1e-6;
    double absolute_tolerance = 1e-5;
    double relative_tolerance = 1e-3;
    std::uint64_t projection_seed = 0;
};

struct GradcheckResult {
    bool passed = true;
    std::string report;
};

using GradcheckFunction = std::function<Tensor(const std::vector<Tensor>&)>;

[[nodiscard]] GradcheckResult gradcheck(const GradcheckFunction& function,
                                        const std::vector<Tensor>& inputs,
                                        const GradcheckOptions& options = {});

} // namespace kael::autograd
