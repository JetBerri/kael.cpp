#pragma once

#include <memory>
#include <optional>

#include "kael/tensor.h"

namespace kael::autograd {

class Node;

struct AutogradMeta {
    bool requires_grad = false;
    std::optional<Tensor> grad;
    std::shared_ptr<Node> grad_fn;
    std::weak_ptr<Node> accumulator;
};

} // namespace kael::autograd
