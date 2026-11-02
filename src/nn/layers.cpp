#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include "kael/factory.h"
#include "kael/nn/dropout.h"
#include "kael/nn/embedding.h"
#include "kael/nn/functional.h"
#include "kael/nn/init.h"
#include "kael/nn/layernorm.h"
#include "kael/nn/linear.h"
#include "kael/nn/sequential.h"
#include "kael/ops.h"

namespace kael::nn {

namespace {

Tensor default_linear_weight(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor weight = empty({out_features, in_features}, dtype);
    init::kaiming_uniform_(weight, std::sqrt(5.0));
    return weight;
}

Tensor default_linear_bias(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor bias = empty({out_features}, dtype);
    const double limit = in_features > 0 ? 1.0 / std::sqrt(static_cast<double>(in_features)) : 0.0;
    return init::uniform_(bias, -limit, limit);
}

} // namespace

Linear::Linear(std::int64_t in_features, std::int64_t out_features, bool bias, DType dtype)
    : weight_(
          register_parameter("weight", default_linear_weight(in_features, out_features, dtype))) {
    if (bias) {
        bias_ = register_parameter("bias", default_linear_bias(in_features, out_features, dtype));
    }
}

Tensor Linear::forward(const Tensor& input) {
    const Tensor product = matmul(input, weight_.T());
    return bias_ ? product + *bias_ : product;
}

Dropout::Dropout(double p) : p_(p) {
    KAEL_CHECK(p >= 0.0 && p < 1.0, "dropout probability must be in [0, 1), got " << p);
}

Tensor Dropout::forward(const Tensor& input) {
    return functional::dropout(input, p_, is_training());
}

LayerNorm::LayerNorm(Shape normalized_shape, double eps, DType dtype)
    : normalized_shape_(std::move(normalized_shape)), eps_(eps),
      weight_(register_parameter("weight", ones(normalized_shape_, dtype))),
      bias_(register_parameter("bias", zeros(normalized_shape_, dtype))) {}

Tensor LayerNorm::forward(const Tensor& input) {
    const std::size_t count = normalized_shape_.size();
    KAEL_CHECK(static_cast<std::size_t>(input.dim()) >= count &&
                   Shape(input.shape().end() - static_cast<std::ptrdiff_t>(count),
                         input.shape().end()) == normalized_shape_,
               "LayerNorm over " << to_string(normalized_shape_) << " got input shape "
                                 << to_string(input.shape()));
    return kael::standardize(input, static_cast<std::int64_t>(count), eps_) * weight_ + bias_;
}

Embedding::Embedding(std::int64_t count, std::int64_t dimension, DType dtype)
    : weight_(register_parameter("weight", empty({count, dimension}, dtype))) {
    init::normal_(weight_, 0.0, 1.0);
}

Tensor Embedding::forward(const Tensor& indices) {
    Shape output_shape = indices.shape();
    output_shape.push_back(weight_.size(1));
    return index_select(weight_, 0, indices.flatten()).reshape(output_shape);
}

Sequential::Sequential(std::initializer_list<std::shared_ptr<UnaryModule>> layers) {
    for (const auto& layer : layers) {
        add(layer);
    }
}

void Sequential::add(std::shared_ptr<UnaryModule> layer) {
    layers_.push_back(register_module(std::to_string(layers_.size()), std::move(layer)));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

} // namespace kael::nn
