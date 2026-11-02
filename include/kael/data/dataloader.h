#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>

#include "kael/data/dataset.h"

namespace kael::data {

struct DataLoaderOptions {
    std::int64_t batch_size = 1;
    bool shuffle = false;
    bool drop_last = false;
};

class DataLoader {
public:
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Example;
        using difference_type = std::ptrdiff_t;
        using pointer = Example*;
        using reference = Example&;

        Iterator(const DataLoader* loader, std::optional<Tensor> order, std::int64_t batch);

        [[nodiscard]] Example& operator*();
        Iterator& operator++();
        [[nodiscard]] bool operator==(const Iterator& other) const noexcept {
            return batch_ == other.batch_;
        }

    private:
        const DataLoader* loader_;
        std::optional<Tensor> order_;
        std::int64_t batch_;
        std::optional<Example> current_;
    };

    DataLoader(const Dataset& dataset, const DataLoaderOptions& options);

    [[nodiscard]] Iterator begin() const;
    [[nodiscard]] Iterator end() const;
    [[nodiscard]] std::int64_t num_batches() const noexcept;

private:
    const Dataset& dataset_;
    DataLoaderOptions options_;
};

} // namespace kael::data
