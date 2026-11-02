#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "kael/nn/module.h"

namespace kael::nn {

class Sequential : public UnaryModule {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<UnaryModule>> layers);

    void add(std::shared_ptr<UnaryModule> layer);
    Tensor forward(const Tensor& input) override;

    [[nodiscard]] std::size_t size() const noexcept { return layers_.size(); }

private:
    std::vector<std::shared_ptr<UnaryModule>> layers_;
};

class Flatten : public UnaryModule {
public:
    explicit Flatten(std::int64_t start_dim = 1, std::int64_t end_dim = -1)
        : start_dim_(start_dim), end_dim_(end_dim) {}
    Tensor forward(const Tensor& input) override { return input.flatten(start_dim_, end_dim_); }

private:
    std::int64_t start_dim_;
    std::int64_t end_dim_;
};

} // namespace kael::nn
