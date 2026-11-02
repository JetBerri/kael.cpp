#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "kael/tensor.h"

namespace kael::io {

using TensorMap = std::map<std::string, Tensor>;
using Metadata = std::map<std::string, std::string>;

enum class LoadMode { Copy, Mapped };

struct SafetensorsFile {
    TensorMap tensors;
    Metadata metadata;
};

void save_safetensors(const std::filesystem::path& path, const TensorMap& tensors,
                      const Metadata& metadata = {});
[[nodiscard]] SafetensorsFile read_safetensors(const std::filesystem::path& path,
                                               LoadMode mode = LoadMode::Copy);
[[nodiscard]] TensorMap load_safetensors(const std::filesystem::path& path,
                                         LoadMode mode = LoadMode::Copy);

} // namespace kael::io
