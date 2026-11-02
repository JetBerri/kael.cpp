#pragma once

#include <cstdint>

#include "kael/dtype.h"
#include "kael/nn/module.h"

namespace kael::nn {

class Embedding : public UnaryModule {
public:
    Embedding(std::int64_t count, std::int64_t dimension, DType dtype = DType::F32);
    Tensor forward(const Tensor& indices) override;

    [[nodiscard]] const Tensor& weight() const noexcept { return weight_; }

private:
    Tensor weight_;
};

} // namespace kael::nn
