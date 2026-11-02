#pragma once

#include "kael/dtype.h"
#include "kael/nn/module.h"
#include "kael/shape.h"

namespace kael::nn {

class LayerNorm : public UnaryModule {
public:
    explicit LayerNorm(Shape normalized_shape, double eps = 1e-5, DType dtype = DType::F32);
    Tensor forward(const Tensor& input) override;

private:
    Shape normalized_shape_;
    double eps_;
    Tensor weight_;
    Tensor bias_;
};

} // namespace kael::nn
