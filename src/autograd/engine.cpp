#include "kael/autograd/engine.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kael/autograd/grad_mode.h"
#include "kael/autograd/node.h"
#include "kael/factory.h"
#include "kael/ops.h"

namespace kael::autograd {

namespace {

Tensor seed_gradient(const Tensor& root, const std::optional<Tensor>& gradient) {
    if (!gradient) {
        KAEL_CHECK_VALUE(root.numel() == 1,
                         "grad can be implicitly created only for scalar outputs, got "
                         "shape "
                             << to_string(root.shape()));
        return ones_like(root);
    }
    KAEL_CHECK_VALUE(gradient->shape() == root.shape(), "gradient shape "
                                                            << to_string(gradient->shape())
                                                            << " does not match output shape "
                                                            << to_string(root.shape()));
    return gradient->to(root.dtype());
}

std::unordered_map<Node*, std::int64_t> count_dependencies(Node* root) {
    std::unordered_map<Node*, std::int64_t> dependencies;
    std::unordered_set<Node*> visited{root};
    std::vector<Node*> pending{root};
    while (!pending.empty()) {
        const Node* node = pending.back();
        pending.pop_back();
        for (const NodeInput& input : node->inputs()) {
            Node* next = input.edge.get();
            if (next == nullptr) {
                continue;
            }
            ++dependencies[next];
            if (visited.insert(next).second) {
                pending.push_back(next);
            }
        }
    }
    return dependencies;
}

void check_gradient(const Node& node, std::size_t index, const Tensor& gradient) {
    const NodeInput& input = node.inputs()[index];
    KAEL_CHECK(gradient.shape() == input.shape,
               "Function " << node.name() << " returned an invalid gradient at index " << index
                           << " - got " << to_string(gradient.shape()) << " but expected shape "
                           << to_string(input.shape));
    KAEL_CHECK(gradient.dtype() == input.dtype,
               "Function " << node.name() << " returned an invalid gradient at index " << index
                           << " - got dtype " << gradient.dtype() << " but expected "
                           << input.dtype);
}

void add_to_buffer(std::unordered_map<Node*, Tensor>& buffers, Node* node, const Tensor& gradient) {
    const auto [slot, inserted] = buffers.try_emplace(node, gradient);
    if (!inserted) {
        slot->second = slot->second + gradient;
    }
}

} // namespace

void backward(const Tensor& root, const std::optional<Tensor>& gradient, bool retain_graph) {
    KAEL_CHECK(root.requires_grad(), "backward needs a tensor that requires grad");
    const NoGradGuard guard;

    const Edge start = gradient_edge(root);
    auto dependencies = count_dependencies(start.get());
    std::unordered_map<Node*, Tensor> buffers;
    buffers.try_emplace(start.get(), seed_gradient(root, gradient));
    std::vector<Node*> ready{start.get()};

    while (!ready.empty()) {
        Node* node = ready.back();
        ready.pop_back();

        GradientList gradients;
        if (const auto buffered = buffers.find(node); buffered != buffers.end()) {
            const Tensor incoming = std::move(buffered->second);
            buffers.erase(buffered);
            node->store_retained_gradient(incoming);
            gradients = node->apply(incoming);
        }
        if (!retain_graph) {
            node->release_variables();
        }

        const std::vector<NodeInput>& inputs = node->inputs();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            Node* next = inputs[i].edge.get();
            if (next == nullptr) {
                continue;
            }
            if (i < gradients.size() && gradients[i]) {
                check_gradient(*node, i, *gradients[i]);
                add_to_buffer(buffers, next, *gradients[i]);
            }
            if (--dependencies[next] == 0) {
                ready.push_back(next);
            }
        }
    }
}

} // namespace kael::autograd
