#include "kael/storage.h"

#include "kael/allocator.h"
#include "kael/error.h"

namespace kael {

void Storage::BlockDeleter::operator()(std::byte* ptr) const noexcept {
    detail::release_block(ptr, capacity);
}

Storage::Storage(std::int64_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    KAEL_CHECK_VALUE(nbytes >= 0, "storage size must be non-negative, got " << nbytes << " bytes");
    KAEL_CHECK(device == Device::CPU, "CUDA storage is not implemented yet");

    if (nbytes > 0) {
        const std::int64_t capacity = detail::block_capacity(nbytes);
        data_ = {detail::allocate_block(capacity), BlockDeleter{capacity}};
    }
}

} // namespace kael
