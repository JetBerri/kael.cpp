#include <cstring>
#include <new>
#include <type_traits>

#include "kael/device_kernels.h"

#include "internal/element_functions.h"

namespace kael::device {

namespace {

// Backwards, so that a kernel relying on the order of its elements fails here too.
template <typename Task> void for_each(std::int64_t count, const Task& task) {
    for (std::int64_t i = count; i-- > 0;) {
        task(i);
    }
}

template <typename T> void atomic_add(T* target, T value) {
    *target = static_cast<T>(functions::wrapping_add(*target, value));
}

} // namespace

} // namespace kael::device

namespace kael::device::tasks {
using device::atomic_add;
}

#include "kernels.inl"

namespace kael::device {

bool available() {
    return true;
}

void synchronize() {}

std::byte* allocate(std::int64_t bytes) {
    return static_cast<std::byte*>(::operator new(static_cast<std::size_t>(bytes)));
}

void release(std::byte* block) noexcept {
    ::operator delete(block);
}

void copy_to_device(std::byte* target, const std::byte* source, std::int64_t bytes) {
    std::memcpy(target, source, static_cast<std::size_t>(bytes));
}

void copy_to_host(std::byte* target, const std::byte* source, std::int64_t bytes) {
    std::memcpy(target, source, static_cast<std::size_t>(bytes));
}

void copy_within_device(std::byte* target, const std::byte* source, std::int64_t bytes) {
    std::memcpy(target, source, static_cast<std::size_t>(bytes));
}

void matmul(DType dtype, const std::byte* a, const std::byte* b, std::byte* out,
            const MatmulSizes& sizes) {
    const std::int64_t tile_rows = (sizes.rows + tasks::kMatmulTile - 1) / tasks::kMatmulTile;
    const std::int64_t tile_columns = (sizes.columns + tasks::kMatmulTile - 1) / tasks::kMatmulTile;
    const std::int64_t tiles = sizes.batches * tile_rows * tile_columns;
    with_type(dtype, [&](auto sample) {
        using T = decltype(sample);
        for_each(tiles,
                 tasks::MatmulTiles<T>{reinterpret_cast<const T*>(a), reinterpret_cast<const T*>(b),
                                       reinterpret_cast<T*>(out), sizes, tile_rows, tile_columns});
    });
}

} // namespace kael::device
