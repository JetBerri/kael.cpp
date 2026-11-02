#include "kael/debug.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "kael/autograd/engine.h"
#include "kael/autograd/node.h"

namespace kael {

namespace {

const char* yes_or_no(bool value) {
    return value ? "yes" : "no";
}

} // namespace

std::string explain(const Tensor& t) {
    std::ostringstream text;
    const auto& impl = *t.impl();
    text << "shape: " << to_string(t.shape()) << "\n"
         << "strides: " << to_string(t.strides()) << "\n"
         << "offset: " << t.offset() << "\n"
         << "dtype: " << t.dtype() << "\n"
         << "device: " << t.device() << "\n"
         << "contiguous: " << yes_or_no(t.is_contiguous()) << "\n"
         << "storage: " << static_cast<const void*>(impl.storage->data()) << ", "
         << impl.storage->nbytes() << " bytes, version " << impl.storage->version()
         << ", references: " << impl.storage.use_count() << "\n"
         << "requires_grad: " << yes_or_no(t.requires_grad()) << "\n"
         << "leaf: " << yes_or_no(t.is_leaf()) << "\n"
         << "has grad: " << yes_or_no(t.grad().has_value()) << "\n";
    if (const auto function = t.grad_fn()) {
        text << "grad_fn: " << function->name() << "\n";
    }
    return text.str();
}

std::string autograd_graph_dot(const Tensor& t) {
    std::ostringstream dot;
    dot << "digraph autograd {\n";
    const autograd::Edge root = t.requires_grad() ? autograd::gradient_edge(t) : nullptr;
    if (root) {
        std::unordered_map<const autograd::Node*, std::size_t> ids{{root.get(), 0}};
        std::vector<const autograd::Node*> pending{root.get()};
        dot << "  n0 [label=\"" << root->name() << "\"];\n";
        while (!pending.empty()) {
            const autograd::Node* node = pending.back();
            pending.pop_back();
            const std::size_t from = ids.at(node);
            for (const autograd::NodeInput& input : node->inputs()) {
                const autograd::Node* next = input.edge.get();
                if (next == nullptr) {
                    continue;
                }
                const auto [slot, inserted] = ids.try_emplace(next, ids.size());
                if (inserted) {
                    dot << "  n" << slot->second << " [label=\"" << next->name() << "\"];\n";
                    pending.push_back(next);
                }
                dot << "  n" << from << " -> n" << slot->second << " [label=\""
                    << to_string(input.shape) << "\"];\n";
            }
        }
    }
    dot << "}\n";
    return dot.str();
}

void set_backward_trace(std::ostream* output) {
    autograd::detail::trace_output() = output;
}

void set_anomaly_detection(bool enabled) {
    autograd::detail::anomaly_detection() = enabled;
}

} // namespace kael
