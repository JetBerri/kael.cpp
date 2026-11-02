#pragma once

#include <cstdint>
#include <functional>

namespace kael {

void set_num_threads(std::int64_t count);
[[nodiscard]] std::int64_t get_num_threads();

inline constexpr std::int64_t kElementGrain = 524288;

using ParallelBody = std::function<void(std::int64_t, std::int64_t)>;

void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const ParallelBody& body);

} // namespace kael
