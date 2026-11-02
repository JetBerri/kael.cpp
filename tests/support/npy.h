#pragma once

#include <filesystem>

#include "kael/tensor.h"

namespace kael::test {

Tensor read_npy(const std::filesystem::path& path);

} // namespace kael::test
