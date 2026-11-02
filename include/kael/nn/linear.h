#pragma once

#include <cstdint>
#include <optional>

#include "kael/dtype.h"
#include "kael/nn/module.h"

namespace kael::nn {

class Linear : public UnaryModule {
public:
    Linear(std::int64_t in_features, std::int64_t out_features, bool bias = true,
           DType dtype = DType::F32);

    Tensor forward(const Tensor& input) override;

    [[nodiscard]] const Tensor& weight() const noexcept { return weight_; }
    [[nodiscard]] const std::optional<Tensor>& bias() const noexcept { return bias_; }

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
};

} // namespace kael::nn
