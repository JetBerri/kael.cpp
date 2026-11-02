#pragma once

#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "kael/autograd/node.h"
#include "kael/tensor.h"

namespace kael::autograd {

class SavedTensor {
public:
    explicit SavedTensor(const Tensor& t) : data_(t.detach()), version_(t.version()) {}

    [[nodiscard]] const Tensor& get() const {
        KAEL_CHECK(data_.version() == version_,
                   "one of the variables needed for gradient computation has been modified by an "
                   "inplace operation: expected version "
                       << version_ << ", got " << data_.version());
        return data_;
    }

private:
    Tensor data_;
    std::int64_t version_;
};

[[nodiscard]] bool should_record(const Tensor& output, std::initializer_list<const Tensor*> inputs);
[[nodiscard]] bool should_record(const Tensor& output, std::span<const Tensor> inputs);

void set_history(const Tensor& output, std::string name, const std::vector<Tensor>& inputs,
                 std::vector<GradientFn> gradients);

} // namespace kael::autograd
