#include "recording.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#include "kael/autograd/functions.h"
#include "kael/autograd/grad_mode.h"
#include "kael/ops.h"

namespace kael::autograd {

namespace {

bool can_record(const Tensor& output) {
    return GradMode::is_enabled() && is_floating_point(output.dtype());
}

} // namespace

bool should_record(const Tensor& output, std::initializer_list<const Tensor*> inputs) {
    return can_record(output) && std::any_of(inputs.begin(), inputs.end(), [](const Tensor* input) {
               return input->requires_grad();
           });
}

bool should_record(const Tensor& output, std::span<const Tensor> inputs) {
    return can_record(output) && std::any_of(inputs.begin(), inputs.end(), [](const Tensor& input) {
               return input.requires_grad();
           });
}

void set_history(const Tensor& output, std::string name, const std::vector<Tensor>& inputs,
                 std::vector<GradientFn> gradients) {
    std::vector<NodeInput> node_inputs;
    node_inputs.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        node_inputs.push_back(make_input(inputs[i]));
        if (!inputs[i].requires_grad()) {
            gradients[i] = nullptr;
        }
    }
    auto meta = std::make_unique<AutogradMeta>();
    meta->requires_grad = true;
    meta->grad_fn = std::make_shared<FunctionNode>(std::move(name), std::move(node_inputs),
                                                   std::move(gradients));
    output.impl()->autograd = std::move(meta);
}

Tensor sum_to_shape(const Tensor& gradient, const Shape& target) {
    KAEL_CHECK(gradient.dim() >= static_cast<std::int64_t>(target.size()),
               "cannot reduce gradient of shape " << to_string(gradient.shape()) << " to "
                                                  << to_string(target));
    const auto extra = static_cast<std::size_t>(gradient.dim()) - target.size();
    Tensor reduced = gradient;
    if (extra > 0) {
        Dims leading(extra);
        std::iota(leading.begin(), leading.end(), std::int64_t{0});
        reduced = kael::sum(reduced, leading);
    }
    Dims stretched;
    for (std::size_t i = 0; i < target.size(); ++i) {
        if (target[i] == 1 && reduced.shape()[i] != 1) {
            stretched.push_back(static_cast<std::int64_t>(i));
        }
    }
    return stretched.empty() ? reduced : kael::sum(reduced, stretched, true);
}

} // namespace kael::autograd
