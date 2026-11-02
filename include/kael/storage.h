#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "kael/dtype.h"

namespace kael {

class Storage {
public:
    static constexpr std::size_t kAlignment = 64;

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
    [[nodiscard]] std::int64_t version() const noexcept { return version_; }
    void bump_version() noexcept { ++version_; }

private:
    struct AlignedDeleter {
        void operator()(std::byte* ptr) const noexcept;
    };

    std::unique_ptr<std::byte[], AlignedDeleter> data_;
    std::int64_t nbytes_;
    Device device_;
    std::int64_t version_ = 0;
};

} // namespace kael
