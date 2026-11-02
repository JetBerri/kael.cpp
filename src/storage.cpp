#include "kael/storage.h"

#include <new>

#include "kael/error.h"

namespace kael {

void Storage::AlignedDeleter::operator()(std::byte* ptr) const noexcept {
    // The alignment must match the one used in the allocation.
    ::operator delete(ptr, std::align_val_t{kAlignment});
}

Storage::Storage(std::int64_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    KAEL_CHECK(nbytes >= 0, "storage size must be non-negative, got " << nbytes << " bytes");
    KAEL_CHECK(device == Device::CPU, "CUDA storage is not implemented yet");

    // Allocating 0 bytes has implementation-defined behaviour: keep it null.
    if (nbytes == 0) {
        return;
    }

    void* raw = ::operator new(static_cast<std::size_t>(nbytes), std::align_val_t{kAlignment});
    data_.reset(static_cast<std::byte*>(raw));
}

} // namespace kael
