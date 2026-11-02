#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "kael/tensor.h"

namespace kael::io {

using TensorMap = std::map<std::string, Tensor>;
using Metadata = std::map<std::string, std::string>;

struct SafetensorsFile {
    TensorMap tensors;
    Metadata metadata;
};

void save_safetensors(const std::filesystem::path& path, const TensorMap& tensors,
                      const Metadata& metadata = {});
[[nodiscard]] SafetensorsFile read_safetensors(const std::filesystem::path& path);
[[nodiscard]] TensorMap load_safetensors(const std::filesystem::path& path);

} // namespace kael::io
