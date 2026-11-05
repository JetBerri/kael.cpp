#pragma once

#include <string_view>

#include "kael/dtype.h"

namespace kael {

[[nodiscard]] Device parse_device(std::string_view name);
[[nodiscard]] bool cuda_is_available();
void cuda_synchronize();

} // namespace kael
