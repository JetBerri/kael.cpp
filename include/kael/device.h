#pragma once

namespace kael {

[[nodiscard]] bool cuda_is_available();
void cuda_synchronize();

} // namespace kael
