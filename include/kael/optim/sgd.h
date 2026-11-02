#pragma once

#include <optional>
#include <vector>

#include "kael/optim/optimizer.h"

namespace kael::optim {

struct SgdOptions {
    double learning_rate = 1e-3;
    double momentum = 0.0;
    double weight_decay = 0.0;
    bool nesterov = false;
};

class SGD final : public Optimizer {
public:
    SGD(std::vector<Tensor> parameters, const SgdOptions& options);

private:
    void update() override;

    SgdOptions options_;
    std::vector<std::optional<Tensor>> velocities_;
};

} // namespace kael::optim
