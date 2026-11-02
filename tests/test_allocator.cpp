#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kael/allocator.h"
#include "kael/kael.h"
#include "kael/storage.h"

namespace {

class CachingAllocator : public testing::Test {
protected:
    void SetUp() override {
        if (!kael::caching_allocator_enabled()) {
            GTEST_SKIP() << "built without the caching allocator";
        }
        kael::empty_cache();
    }
    void TearDown() override { kael::empty_cache(); }
};

} // namespace

TEST_F(CachingAllocator, SameSizeReusesTheBlock) {
    std::byte* first = nullptr;
    {
        kael::Storage storage(1000);
        first = storage.data();
    }
    const kael::AllocatorStats before = kael::allocator_stats();
    const kael::Storage storage(1000);
    const kael::AllocatorStats after = kael::allocator_stats();
    EXPECT_EQ(storage.data(), first);
    EXPECT_EQ(after.hits, before.hits + 1);
    EXPECT_EQ(after.misses, before.misses);
}

TEST_F(CachingAllocator, SizesInTheSameBucketShareBlocks) {
    EXPECT_EQ(kael::detail::block_capacity(1), 512);
    EXPECT_EQ(kael::detail::block_capacity(512), 512);
    EXPECT_EQ(kael::detail::block_capacity(513), 1024);
    EXPECT_EQ(kael::detail::block_capacity(65536), 65536);
    EXPECT_EQ(kael::detail::block_capacity(65537), 131072);
    EXPECT_EQ(kael::detail::block_capacity(200000), 262144);
}

TEST_F(CachingAllocator, DifferentBucketIsAMiss) {
    {
        const kael::Storage storage(1000);
    }
    const kael::AllocatorStats before = kael::allocator_stats();
    const kael::Storage storage(5000);
    EXPECT_EQ(kael::allocator_stats().misses, before.misses + 1);
}

TEST_F(CachingAllocator, ReusedBlocksStayAligned) {
    for (const std::int64_t bytes : {1, 700, 4096, 70000}) {
        {
            const kael::Storage storage(bytes);
        }
        const kael::Storage storage(bytes);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage.data()) % kael::Storage::kAlignment, 0u);
    }
}

TEST_F(CachingAllocator, StatsFollowTheBlockLifetime) {
    const kael::AllocatorStats start = kael::allocator_stats();
    EXPECT_EQ(start.cached_bytes, 0);
    {
        const kael::Storage storage(1000);
        const kael::AllocatorStats live = kael::allocator_stats();
        EXPECT_EQ(live.in_use_bytes, start.in_use_bytes + 1024);
        EXPECT_EQ(live.reserved_bytes, start.reserved_bytes + 1024);
    }
    const kael::AllocatorStats released = kael::allocator_stats();
    EXPECT_EQ(released.in_use_bytes, start.in_use_bytes);
    EXPECT_EQ(released.cached_bytes, 1024);
    EXPECT_EQ(released.reserved_bytes, start.reserved_bytes + 1024);
}

TEST_F(CachingAllocator, EmptyCacheGivesMemoryBack) {
    const kael::AllocatorStats start = kael::allocator_stats();
    {
        const kael::Storage storage(1000);
    }
    kael::empty_cache();
    const kael::AllocatorStats emptied = kael::allocator_stats();
    EXPECT_EQ(emptied.cached_bytes, 0);
    EXPECT_EQ(emptied.reserved_bytes, start.reserved_bytes);

    const kael::AllocatorStats before = kael::allocator_stats();
    const kael::Storage storage(1000);
    EXPECT_EQ(kael::allocator_stats().misses, before.misses + 1);
}

TEST_F(CachingAllocator, HugeBlocksAreNotKeptWhenOverTheLimit) {
    const std::int64_t huge = std::int64_t{1} << 31;
    {
        const kael::Storage storage(huge);
    }
    EXPECT_EQ(kael::allocator_stats().cached_bytes, 0);
}

TEST_F(CachingAllocator, DisablingBypassesTheCache) {
    {
        const kael::Storage storage(1000);
    }
    kael::set_caching_allocator_enabled(false);
    EXPECT_EQ(kael::allocator_stats().cached_bytes, 0);
    const kael::AllocatorStats before = kael::allocator_stats();
    {
        const kael::Storage storage(1000);
    }
    const kael::AllocatorStats after = kael::allocator_stats();
    kael::set_caching_allocator_enabled(true);
    EXPECT_EQ(after.hits, before.hits);
    EXPECT_EQ(after.cached_bytes, 0);
    EXPECT_EQ(after.in_use_bytes, before.in_use_bytes);
}

TEST_F(CachingAllocator, BlockDisabledMidLifeIsFreedCleanly) {
    auto storage = std::make_unique<kael::Storage>(1000);
    kael::set_caching_allocator_enabled(false);
    storage.reset();
    kael::set_caching_allocator_enabled(true);
    EXPECT_EQ(kael::allocator_stats().cached_bytes, 0);
}

TEST_F(CachingAllocator, RepeatedTensorMathHitsTheCache) {
    const kael::Tensor x = kael::ones({128, 128});
    {
        const kael::Tensor warm = x + x;
    }
    const kael::AllocatorStats before = kael::allocator_stats();
    for (int i = 0; i < 10; ++i) {
        const kael::Tensor y = x + x;
    }
    const kael::AllocatorStats after = kael::allocator_stats();
    EXPECT_EQ(after.hits - before.hits, 10);
    EXPECT_EQ(after.misses, before.misses);
}

TEST(Allocator, ConcurrentUseKeepsTheBooksBalanced) {
    const std::int64_t in_use = kael::allocator_stats().in_use_bytes;
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < 2000; ++i) {
                kael::Storage storage((i % 17 + 1) * 700 + t);
                storage.data()[0] = std::byte{1};
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(kael::allocator_stats().in_use_bytes, in_use);
}

TEST(Allocator, ZeroBytesStayNull) {
    const kael::Storage storage(0);
    EXPECT_EQ(storage.data(), nullptr);
}

TEST_F(CachingAllocator, FailedReservationLeavesTheBooksUntouched) {
    const kael::AllocatorStats before = kael::allocator_stats();
    EXPECT_THROW(kael::Storage(std::int64_t{1} << 62), std::bad_alloc);
    const kael::AllocatorStats after = kael::allocator_stats();
    EXPECT_EQ(after.in_use_bytes, before.in_use_bytes);
    EXPECT_EQ(after.reserved_bytes, before.reserved_bytes);
}

TEST(Allocator, SizesWithoutARepresentableBucketAreRejected) {
    EXPECT_EQ(kael::detail::block_capacity(std::int64_t{1} << 62), std::int64_t{1} << 62);
    EXPECT_THROW((void)kael::detail::block_capacity((std::int64_t{1} << 62) + 1), kael::Error);
}
