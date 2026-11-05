#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct Node {
    double data = 0.0;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(Node&)> propagate;
};

class Value {
public:
    explicit Value(double data) : node_(std::make_shared<Node>()) { node_->data = data; }

    [[nodiscard]] double data() const { return node_->data; }
    [[nodiscard]] double grad() const { return node_->grad; }

    friend Value operator+(const Value& a, const Value& b) {
        return Value::derived(a.data() + b.data(), {a.node_, b.node_}, [](Node& out) {
            out.parents[0]->grad += out.grad;
            out.parents[1]->grad += out.grad;
        });
    }

    friend Value operator*(const Value& a, const Value& b) {
        return Value::derived(a.data() * b.data(), {a.node_, b.node_}, [](Node& out) {
            out.parents[0]->grad += out.parents[1]->data * out.grad;
            out.parents[1]->grad += out.parents[0]->data * out.grad;
        });
    }

    [[nodiscard]] Value tanh() const {
        return Value::derived(std::tanh(data()), {node_}, [](Node& out) {
            out.parents[0]->grad += (1.0 - out.data * out.data) * out.grad;
        });
    }

    [[nodiscard]] Value exp() const {
        return Value::derived(std::exp(data()), {node_},
                              [](Node& out) { out.parents[0]->grad += out.data * out.grad; });
    }

    void backward() const {
        std::vector<Node*> order;
        std::unordered_set<Node*> visited;
        std::function<void(Node*)> visit = [&](Node* node) {
            if (!visited.insert(node).second) {
                return;
            }
            for (const auto& parent : node->parents) {
                visit(parent.get());
            }
            order.push_back(node);
        };
        visit(node_.get());

        node_->grad = 1.0;
        std::for_each(order.rbegin(), order.rend(), [](Node* node) {
            if (node->propagate) {
                node->propagate(*node);
            }
        });
    }

private:
    static Value derived(double data, std::vector<std::shared_ptr<Node>> parents,
                         std::function<void(Node&)> propagate) {
        Value out(data);
        out.node_->parents = std::move(parents);
        out.node_->propagate = std::move(propagate);
        return out;
    }

    std::shared_ptr<Node> node_;
};

bool close(double actual, double expected) {
    return std::abs(actual - expected) < 1e-12;
}

bool neuron_matches_hand_derivation() {
    const double w1 = 0.5, x1 = 2.0, w2 = -1.5, x2 = 0.5, b = 0.25;
    const double activation = w1 * x1 + w2 * x2 + b;
    const double slope = 1.0 - std::tanh(activation) * std::tanh(activation);

    const Value vw1(w1), vx1(x1), vw2(w2), vx2(x2), vb(b);
    const Value y = (vw1 * vx1 + vw2 * vx2 + vb).tanh();
    y.backward();

    return close(y.data(), std::tanh(activation)) && close(vw1.grad(), slope * x1) &&
           close(vx1.grad(), slope * w1) && close(vw2.grad(), slope * x2) &&
           close(vx2.grad(), slope * w2) && close(vb.grad(), slope);
}

bool repeated_input_accumulates() {
    const Value a(3.0);
    (a * a).backward();
    return close(a.grad(), 6.0);
}

bool diamond_waits_for_both_paths() {
    const Value x(2.0);
    (x * Value(2.0) + x * Value(3.0)).backward();
    return close(x.grad(), 5.0);
}

bool exp_derivative_is_itself() {
    const Value x(1.5);
    x.exp().backward();
    return close(x.grad(), std::exp(1.5));
}

} // namespace

int main() {
    const bool passed = neuron_matches_hand_derivation() && repeated_input_accumulates() &&
                        diamond_waits_for_both_paths() && exp_derivative_is_itself();
    std::cout << (passed ? "micrograd: gradients match" : "micrograd: gradient mismatch") << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
