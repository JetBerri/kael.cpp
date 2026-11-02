#include "kael/allocator.h"

#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "kael/error.h"
#include "kael/storage.h"

namespace kael {

namespace {

constexpr std::int64_t kBucketStep = 512;
constexpr std::int64_t kLargestSteppedSize = 65536;
constexpr std::int64_t kMaxCachedBytes = std::int64_t{1} << 30;
constexpr std::int64_t kLargestBucket = std::int64_t{1} << 62;

#if defined(KAEL_NO_CACHING_ALLOCATOR)
constexpr bool kCachingCompiledIn = false;
#else
constexpr bool kCachingCompiledIn = true;
#endif

std::byte* system_allocate(std::int64_t capacity) {
    return static_cast<std::byte*>(
        ::operator new(static_cast<std::size_t>(capacity), std::align_val_t{Storage::kAlignment}));
}

void system_free(std::byte* block) noexcept {
    ::operator delete(block, std::align_val_t{Storage::kAlignment});
}

class BlockCache {
public:
    std::byte* allocate(std::int64_t capacity) {
        if (std::byte* cached = take_cached(capacity)) {
            return cached;
        }
        std::byte* block = system_allocate(capacity);
        const std::lock_guard lock(mutex_);
        stats_.in_use_bytes += capacity;
        stats_.reserved_bytes += capacity;
        ++stats_.misses;
        return block;
    }

    void release(std::byte* block, std::int64_t capacity) noexcept {
        {
            const std::lock_guard lock(mutex_);
            stats_.in_use_bytes -= capacity;
            if (enabled_ && stats_.cached_bytes + capacity <= kMaxCachedBytes) {
                free_blocks_[capacity].push_back(block);
                stats_.cached_bytes += capacity;
                return;
            }
            stats_.reserved_bytes -= capacity;
        }
        system_free(block);
    }

    void clear() noexcept {
        const std::lock_guard lock(mutex_);
        drop_free_blocks();
    }

    [[nodiscard]] AllocatorStats stats() {
        const std::lock_guard lock(mutex_);
        return stats_;
    }

    [[nodiscard]] bool enabled() {
        const std::lock_guard lock(mutex_);
        return enabled_;
    }

    void set_enabled(bool enabled) {
        const std::lock_guard lock(mutex_);
        enabled_ = enabled;
        if (!enabled_) {
            drop_free_blocks();
        }
    }

private:
    std::byte* take_cached(std::int64_t capacity) {
        const std::lock_guard lock(mutex_);
        auto& blocks = free_blocks_[capacity];
        if (blocks.empty()) {
            return nullptr;
        }
        std::byte* block = blocks.back();
        blocks.pop_back();
        stats_.in_use_bytes += capacity;
        stats_.cached_bytes -= capacity;
        ++stats_.hits;
        return block;
    }

    static bool enabled_at_startup() {
        return kCachingCompiledIn && std::getenv("KAEL_NO_CACHING_ALLOCATOR") == nullptr;
    }

    void drop_free_blocks() noexcept {
        for (auto& [capacity, blocks] : free_blocks_) {
            for (std::byte* block : blocks) {
                system_free(block);
            }
            stats_.reserved_bytes -= capacity * static_cast<std::int64_t>(blocks.size());
            stats_.cached_bytes -= capacity * static_cast<std::int64_t>(blocks.size());
            blocks.clear();
        }
    }

    std::mutex mutex_;
    std::unordered_map<std::int64_t, std::vector<std::byte*>> free_blocks_;
    AllocatorStats stats_;
    bool enabled_ = enabled_at_startup();
};

// Never destroyed: tensors with static storage can release their blocks after main returns.
BlockCache& block_cache() {
    static auto* cache = new BlockCache();
    return *cache;
}

std::int64_t next_power_of_two(std::int64_t value) {
    std::int64_t power = kLargestSteppedSize;
    while (power < value) {
        power *= 2;
    }
    return power;
}

} // namespace

AllocatorStats allocator_stats() {
    return block_cache().stats();
}

void empty_cache() {
    block_cache().clear();
}

bool caching_allocator_enabled() {
    return block_cache().enabled();
}

void set_caching_allocator_enabled(bool enabled) {
    block_cache().set_enabled(enabled && kCachingCompiledIn);
}

namespace detail {

std::int64_t block_capacity(std::int64_t nbytes) {
    if (nbytes <= kLargestSteppedSize) {
        return (nbytes + kBucketStep - 1) / kBucketStep * kBucketStep;
    }
    KAEL_CHECK(nbytes <= kLargestBucket, "cannot reserve " << nbytes << " bytes");
    return next_power_of_two(nbytes);
}

std::byte* allocate_block(std::int64_t capacity) {
    return block_cache().allocate(capacity);
}

void release_block(std::byte* block, std::int64_t capacity) noexcept {
    block_cache().release(block, capacity);
}

} // namespace detail

} // namespace kael
