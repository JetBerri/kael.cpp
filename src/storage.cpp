#include "kael/storage.h"

#include <new>

#include "kael/error.h"

namespace kael {

void Storage::AlignedDeleter::operator()(std::byte* ptr) const noexcept {
    ::operator delete(ptr, std::align_val_t{kAlignment});
}

Storage::Storage(std::int64_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    KAEL_CHECK(nbytes >= 0, "storage size must be non-negative, got " << nbytes << " bytes");
    KAEL_CHECK(device == Device::CPU, "CUDA storage is not implemented yet");

    if (nbytes > 0) {
        void* memory =
            ::operator new(static_cast<std::size_t>(nbytes), std::align_val_t{kAlignment});
        data_.reset(static_cast<std::byte*>(memory));
    }
}

} // namespace kael
