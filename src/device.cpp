#include "kael/device.h"

#include <string>

#include "kael/device_kernels.h"
#include "kael/error.h"

namespace kael {

Device parse_device(std::string_view name) {
    if (name == "cpu") {
        return Device::CPU;
    }
    if (name == "cuda" || name == "cuda:0") {
        return Device::CUDA;
    }
    throw ValueError("unknown device '" + std::string(name) + "', expected 'cpu' or 'cuda'");
}

bool cuda_is_available() {
    return device::available();
}

void cuda_synchronize() {
    device::synchronize();
}

} // namespace kael
