#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kael/tensor.h"

namespace kael::nn {

using NamedTensor = std::pair<std::string, Tensor>;
using StateDict = std::map<std::string, Tensor>;

class Module {
public:
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    virtual ~Module() = default;

    Tensor register_parameter(std::string name, Tensor parameter);

    template <typename M>
    std::shared_ptr<M> register_module(std::string name, std::shared_ptr<M> m) {
        add_child(std::move(name), m);
        return m;
    }

    [[nodiscard]] std::vector<Tensor> parameters() const;
    [[nodiscard]] std::vector<NamedTensor> named_parameters() const;
    [[nodiscard]] std::int64_t num_parameters() const;

    void train(bool training = true);
    void eval() { train(false); }
    [[nodiscard]] bool is_training() const noexcept { return training_; }

    void zero_grad();
    void to(Device device);

    [[nodiscard]] StateDict state_dict() const;
    void load_state_dict(const StateDict& state);

private:
    void add_child(std::string name, std::shared_ptr<Module> child);
    void collect_parameters(const std::string& prefix, std::vector<NamedTensor>& out) const;

    std::vector<NamedTensor> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};

class UnaryModule : public Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input) { return forward(input); }
};

} // namespace kael::nn
