#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "kael/data/mnist.h"
#include "kael/error.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("kael_fuzz_idx_" + std::to_string(getpid()));
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    try {
        (void)kael::data::read_idx_images(path);
    } catch (const kael::Error&) {
    }
    try {
        (void)kael::data::read_idx_labels(path);
    } catch (const kael::Error&) {
    }
    return 0;
}
