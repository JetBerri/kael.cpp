#include "kael/device.h"

#include "kael/device_kernels.h"

namespace kael {

bool cuda_is_available() {
    return device::available();
}

void cuda_synchronize() {
    device::synchronize();
}

} // namespace kael
