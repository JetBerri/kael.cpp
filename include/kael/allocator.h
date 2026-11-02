#pragma once

#include <cstddef>
#include <cstdint>

#include "kael/dtype.h"

namespace kael {

struct AllocatorStats {
    std::int64_t reserved_bytes = 0;
    std::int64_t in_use_bytes = 0;
    std::int64_t cached_bytes = 0;
    std::int64_t hits = 0;
    std::int64_t misses = 0;
};

[[nodiscard]] AllocatorStats allocator_stats(Device device = Device::CPU);
void empty_cache();

[[nodiscard]] bool caching_allocator_enabled();
void set_caching_allocator_enabled(bool enabled);

namespace detail {

[[nodiscard]] std::int64_t block_capacity(std::int64_t nbytes);
[[nodiscard]] std::byte* allocate_block(std::int64_t capacity, Device device);
void release_block(std::byte* block, std::int64_t capacity, Device device) noexcept;

} // namespace detail

} // namespace kael
