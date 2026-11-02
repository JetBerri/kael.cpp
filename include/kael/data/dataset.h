#pragma once

#include <cstdint>

#include "kael/tensor.h"

namespace kael::data {

struct Example {
    Tensor input;
    Tensor target;
};

class Dataset {
public:
    Dataset() = default;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    virtual ~Dataset() = default;

    [[nodiscard]] virtual std::int64_t size() const = 0;
    [[nodiscard]] virtual Example get(std::int64_t index) const = 0;
    [[nodiscard]] virtual Example get_batch(const Tensor& indices) const;
};

class TensorDataset final : public Dataset {
public:
    TensorDataset(Tensor inputs, Tensor targets);

    [[nodiscard]] std::int64_t size() const override { return inputs_.size(0); }
    [[nodiscard]] Example get(std::int64_t index) const override;
    [[nodiscard]] Example get_batch(const Tensor& indices) const override;

private:
    Tensor inputs_;
    Tensor targets_;
};

} // namespace kael::data
