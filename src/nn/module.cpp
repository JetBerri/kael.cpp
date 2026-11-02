#include "kael/nn/module.h"

#include <unordered_set>

#include "kael/autograd/grad_mode.h"
#include "kael/autograd/meta.h"
#include "kael/error.h"

namespace kael::nn {

Tensor Module::register_parameter(std::string name, Tensor parameter) {
    parameter.requires_grad_();
    parameters_.emplace_back(std::move(name), parameter);
    return parameter;
}

void Module::add_child(std::string name, std::shared_ptr<Module> child) {
    KAEL_CHECK(child != nullptr, "cannot register a null module named " << name);
    children_.emplace_back(std::move(name), std::move(child));
}

void Module::collect_parameters(const std::string& prefix, std::vector<NamedTensor>& out) const {
    for (const auto& [name, parameter] : parameters_) {
        out.emplace_back(prefix + name, parameter);
    }
    for (const auto& [name, child] : children_) {
        child->collect_parameters(prefix + name + ".", out);
    }
}

std::vector<NamedTensor> Module::named_parameters() const {
    std::vector<NamedTensor> all;
    collect_parameters("", all);

    std::unordered_set<const TensorImpl*> seen;
    std::vector<NamedTensor> unique;
    for (auto& entry : all) {
        if (seen.insert(entry.second.impl().get()).second) {
            unique.push_back(std::move(entry));
        }
    }
    return unique;
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> tensors;
    for (const auto& [name, parameter] : named_parameters()) {
        tensors.push_back(parameter);
    }
    return tensors;
}

std::int64_t Module::num_parameters() const {
    std::int64_t total = 0;
    for (const Tensor& parameter : parameters()) {
        total += parameter.numel();
    }
    return total;
}

void Module::train(bool training) {
    training_ = training;
    for (const auto& [name, child] : children_) {
        child->train(training);
    }
}

void Module::zero_grad() {
    for (Tensor& parameter : parameters()) {
        parameter.zero_grad();
    }
}

void Module::to(Device device) {
    const autograd::NoGradGuard guard;
    for (Tensor& parameter : parameters()) {
        if (parameter.device() == device) {
            continue;
        }
        const Tensor moved = parameter.detach().to(device);
        TensorImpl& impl = *parameter.impl();
        impl.storage = moved.impl()->storage;
        impl.strides = moved.strides();
        impl.offset = moved.offset();
        impl.device = device;
        if (const std::optional<Tensor> gradient = parameter.grad()) {
            impl.autograd->grad = gradient->to(device);
        }
    }
}

StateDict Module::state_dict() const {
    std::vector<NamedTensor> all;
    collect_parameters("", all);
    StateDict state;
    for (const auto& [name, parameter] : all) {
        state.emplace(name, parameter.detach());
    }
    return state;
}

void Module::load_state_dict(const StateDict& state) {
    std::vector<NamedTensor> all;
    collect_parameters("", all);
    for (const auto& [name, parameter] : all) {
        const auto found = state.find(name);
        KAEL_CHECK(found != state.end(), "state dict is missing parameter " << name);
        KAEL_CHECK_VALUE(found->second.shape() == parameter.shape(),
                         "parameter " << name << " has shape " << to_string(parameter.shape())
                                      << " but the state dict has "
                                      << to_string(found->second.shape()));
    }
    const autograd::NoGradGuard guard;
    for (auto& [name, parameter] : all) {
        parameter.copy_(state.at(name));
    }
}

} // namespace kael::nn
