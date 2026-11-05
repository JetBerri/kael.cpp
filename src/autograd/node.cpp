#include "kael/autograd/node.h"

#include <utility>

#include "kael/ops.h"

namespace kael::autograd {

namespace {

void accumulate(TensorImpl& target, const Tensor& gradient) {
    if (!target.autograd) {
        return;
    }
    std::optional<Tensor>& stored = target.autograd->grad;
    if (stored) {
        stored->add_(gradient);
    } else {
        stored = gradient.clone();
    }
}

} // namespace

Node::Node(std::vector<NodeInput> inputs) : inputs_(std::move(inputs)) {}

// A long chain would otherwise overflow the stack by destroying each node from the previous one.
Node::~Node() {
    std::vector<Edge> pending;
    move_edges_into(pending);
    while (!pending.empty()) {
        const Edge edge = std::move(pending.back());
        pending.pop_back();
        if (edge && edge.use_count() == 1) {
            edge->move_edges_into(pending);
        }
    }
}

void Node::move_edges_into(std::vector<Edge>& pending) {
    for (NodeInput& input : inputs_) {
        pending.push_back(std::move(input.edge));
    }
    inputs_.clear();
}

void Node::store_retained_gradient(const Tensor& gradient) const {
    if (const auto output = retained_output_.lock()) {
        accumulate(*output, gradient);
    }
}

AccumulateGrad::AccumulateGrad(std::shared_ptr<TensorImpl> leaf)
    : Node({}), leaf_(std::move(leaf)) {}

GradientList AccumulateGrad::apply(const Tensor& gradient) {
    accumulate(*leaf_, gradient);
    return {};
}

FunctionNode::FunctionNode(std::string name, std::vector<NodeInput> inputs,
                           std::vector<GradientFn> gradients)
    : Node(std::move(inputs)), name_(std::move(name)), gradients_(std::move(gradients)) {}

GradientList FunctionNode::apply(const Tensor& gradient) {
    KAEL_CHECK(!released_, "Trying to backward through the graph a second time ("
                               << name_ << "); specify retain_graph=true");
    GradientList results;
    results.reserve(gradients_.size());
    for (const GradientFn& compute : gradients_) {
        results.push_back(compute ? std::optional<Tensor>(compute(gradient)) : std::nullopt);
    }
    return results;
}

void FunctionNode::release_variables() {
    gradients_.clear();
    released_ = true;
}

Edge gradient_edge(const Tensor& t) {
    if (!t.requires_grad()) {
        return nullptr;
    }
    if (Edge function = t.grad_fn()) {
        return function;
    }
    AutogradMeta& meta = *t.impl()->autograd;
    Edge accumulator = meta.accumulator.lock();
    if (!accumulator) {
        accumulator = std::make_shared<AccumulateGrad>(t.impl());
        meta.accumulator = accumulator;
    }
    return accumulator;
}

NodeInput make_input(const Tensor& t) {
    return {gradient_edge(t), t.shape(), t.dtype()};
}

} // namespace kael::autograd
