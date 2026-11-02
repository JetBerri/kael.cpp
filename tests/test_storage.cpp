#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/storage.h"

static_assert(!std::is_copy_constructible_v<kael::Storage>);
static_assert(!std::is_copy_assignable_v<kael::Storage>);
static_assert(std::is_move_constructible_v<kael::Storage>);

TEST(Storage, Allocates) {
    kael::Storage s(100);
    EXPECT_EQ(s.nbytes(), 100);
    EXPECT_NE(s.data(), nullptr);
    EXPECT_EQ(s.device(), kael::Device::CPU);
}

TEST(Storage, AlignedTo64Bytes) {
    for (std::int64_t n : {1, 3, 64, 100, 4097}) {
        kael::Storage s(n);
        const auto address = reinterpret_cast<std::uintptr_t>(s.data());
        EXPECT_EQ(address % kael::Storage::kAlignment, 0u) << "nbytes=" << n;
    }
}

TEST(Storage, WriteAndReadEveryByte) {
    kael::Storage s(256);
    for (std::int64_t i = 0; i < s.nbytes(); ++i) {
        s.data()[i] = static_cast<std::byte>(i);
    }
    const kael::Storage& cs = s;
    for (std::int64_t i = 0; i < cs.nbytes(); ++i) {
        EXPECT_EQ(cs.data()[i], static_cast<std::byte>(i));
    }
}

TEST(Storage, ZeroBytesIsNull) {
    EXPECT_NO_THROW({
        kael::Storage s(0);
        EXPECT_EQ(s.nbytes(), 0);
        EXPECT_EQ(s.data(), nullptr);
    });
}

TEST(Storage, NegativeBytesThrows) {
    EXPECT_THROW(kael::Storage(-1), kael::Error);
}

TEST(Storage, CudaThrows) {
    EXPECT_THROW(kael::Storage(16, kael::Device::CUDA), kael::Error);
}

TEST(Storage, Movable) {
    kael::Storage a(32);
    std::byte* ptr = a.data();
    kael::Storage b(std::move(a));
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.nbytes(), 32);
}
