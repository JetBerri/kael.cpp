#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "kael/error.h"
#include "kael/io/safetensors.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    static const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("kael_fuzz_st_" + std::to_string(getpid()));
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    for (const auto mode : {kael::io::LoadMode::Copy, kael::io::LoadMode::Mapped}) {
        try {
            const auto loaded = kael::io::read_safetensors(path, mode);
            for (const auto& [name, tensor] : loaded.tensors) {
                if (tensor.nbytes() > 0) {
                    volatile auto first = *tensor.data_ptr_raw();
                    volatile auto last = tensor.data_ptr_raw()[tensor.nbytes() - 1];
                    (void)first;
                    (void)last;
                }
            }
        } catch (const kael::Error&) {
        }
    }
    return 0;
}
