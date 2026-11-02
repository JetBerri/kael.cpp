#pragma once

#include <filesystem>

#include "kael/tensor.h"

namespace kael::data {

[[nodiscard]] Tensor read_idx_images(const std::filesystem::path& path);
[[nodiscard]] Tensor read_idx_labels(const std::filesystem::path& path);

struct MnistSplit {
    Tensor images;
    Tensor labels;
};

enum class MnistPart { Train, Test };

[[nodiscard]] MnistSplit load_mnist(const std::filesystem::path& directory, MnistPart part);

} // namespace kael::data
