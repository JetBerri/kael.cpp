#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "kael/dtype.h"

namespace kael {

// A block of raw memory: a number of bytes on a device. It knows nothing
// about shapes, strides or element types; TensorImpl adds that meaning.
//
// - The memory is aligned to kAlignment bytes (a cache line; what SIMD will
//   want in phase 9).
// - The memory is NOT initialized. Factories decide what to write.
// - A zero-byte storage holds a null pointer and allocates nothing.
// - Not copyable: two owners of the same buffer would free it twice. In
//   practice it is always shared through a std::shared_ptr<Storage>.
class Storage {
public:
    static constexpr std::size_t kAlignment = 64;

    // Throws if nbytes < 0 or the device is not CPU.
    explicit Storage(std::int64_t nbytes, Device device = Device::CPU);

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;
    ~Storage() = default;

    [[nodiscard]] std::byte* data() noexcept { return data_.get(); }
    [[nodiscard]] const std::byte* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::int64_t nbytes() const noexcept { return nbytes_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

private:
    // Frees memory obtained with the aligned operator new. Every way of
    // allocating has its matching way of freeing; mixing them is UB.
    struct AlignedDeleter {
        void operator()(std::byte* ptr) const noexcept;
    };

    std::unique_ptr<std::byte[], AlignedDeleter> data_;
    std::int64_t nbytes_;
    Device device_;
};

} // namespace kael
