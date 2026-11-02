#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kael/autograd/meta.h"
#include "kael/tensor.h"

namespace kael::autograd {

using Edge = std::shared_ptr<Node>;
using GradientList = std::vector<std::optional<Tensor>>;
using GradientFn = std::function<Tensor(const Tensor&)>;

struct NodeInput {
    Edge edge;
    Shape shape;
    DType dtype;
};

class Node {
public:
    explicit Node(std::vector<NodeInput> inputs);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual ~Node();

    [[nodiscard]] virtual std::string name() const = 0;
    virtual GradientList apply(const Tensor& gradient) = 0;
    virtual void release_variables() {}

    [[nodiscard]] const std::vector<NodeInput>& inputs() const noexcept { return inputs_; }

    void retain_output(const std::shared_ptr<TensorImpl>& output) { retained_output_ = output; }
    void store_retained_gradient(const Tensor& gradient) const;

private:
    void move_edges_into(std::vector<Edge>& pending);

    std::vector<NodeInput> inputs_;
    std::weak_ptr<TensorImpl> retained_output_;
};

class AccumulateGrad final : public Node {
public:
    explicit AccumulateGrad(std::shared_ptr<TensorImpl> leaf);

    [[nodiscard]] std::string name() const override { return "AccumulateGrad"; }
    GradientList apply(const Tensor& gradient) override;

private:
    std::shared_ptr<TensorImpl> leaf_;
};

class FunctionNode final : public Node {
public:
    FunctionNode(std::string name, std::vector<NodeInput> inputs,
                 std::vector<GradientFn> gradients);

    [[nodiscard]] std::string name() const override { return name_; }
    GradientList apply(const Tensor& gradient) override;
    void release_variables() override;

private:
    std::string name_;
    std::vector<GradientFn> gradients_;
    bool released_ = false;
};

[[nodiscard]] Edge gradient_edge(const Tensor& t);
[[nodiscard]] NodeInput make_input(const Tensor& t);

} // namespace kael::autograd
