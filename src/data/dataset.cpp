#include "kael/data/dataset.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "kael/data/dataloader.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/random.h"

namespace kael::data {

Example Dataset::get_batch(const Tensor& indices) const {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    for (std::int64_t i = 0; i < indices.numel(); ++i) {
        Example example = get(indices.at<std::int64_t>({i}));
        inputs.push_back(std::move(example.input));
        targets.push_back(std::move(example.target));
    }
    return {stack(inputs), stack(targets)};
}

TensorDataset::TensorDataset(Tensor inputs, Tensor targets)
    : inputs_(std::move(inputs)), targets_(std::move(targets)) {
    KAEL_CHECK_VALUE(inputs_.dim() >= 1 && targets_.dim() >= 1 &&
                         inputs_.size(0) == targets_.size(0),
                     "TensorDataset needs the same first dimension, got shapes "
                         << to_string(inputs_.shape()) << " and " << to_string(targets_.shape()));
}

Example TensorDataset::get(std::int64_t index) const {
    return {inputs_.select(0, index), targets_.select(0, index)};
}

Example TensorDataset::get_batch(const Tensor& indices) const {
    return {index_select(inputs_, 0, indices), index_select(targets_, 0, indices)};
}

DataLoader::DataLoader(const Dataset& dataset, const DataLoaderOptions& options)
    : dataset_(dataset), options_(options) {
    KAEL_CHECK_VALUE(options.batch_size > 0,
                     "batch size must be positive, got " << options.batch_size);
}

std::int64_t DataLoader::num_batches() const noexcept {
    const std::int64_t size = dataset_.size();
    return options_.drop_last ? size / options_.batch_size
                              : (size + options_.batch_size - 1) / options_.batch_size;
}

DataLoader::Iterator DataLoader::begin() const {
    const std::int64_t size = dataset_.size();
    Tensor order =
        options_.shuffle ? randperm(size) : arange(0, static_cast<double>(size), 1.0, DType::I64);
    return Iterator(this, std::move(order), 0);
}

DataLoader::Iterator DataLoader::end() const {
    return Iterator(this, std::nullopt, num_batches());
}

DataLoader::Iterator::Iterator(const DataLoader* loader, std::optional<Tensor> order,
                               std::int64_t batch)
    : loader_(loader), order_(std::move(order)), batch_(batch) {}

Example& DataLoader::Iterator::operator*() {
    if (!current_) {
        const std::int64_t batch_size = loader_->options_.batch_size;
        const std::int64_t start = batch_ * batch_size;
        const std::int64_t length = std::min(batch_size, order_->numel() - start);
        current_ = loader_->dataset_.get_batch(order_->narrow(0, start, length));
    }
    return *current_;
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++batch_;
    current_.reset();
    return *this;
}

} // namespace kael::data
