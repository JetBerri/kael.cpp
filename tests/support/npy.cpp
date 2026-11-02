#include "support/npy.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "kael/factory.h"

namespace kael::test {

namespace {

constexpr std::array<char, 6> kMagic = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

std::string header_value(const std::string& header, const std::string& key, char terminator) {
    const std::string marker = "'" + key + "': ";
    const auto marker_position = header.find(marker);
    KAEL_CHECK(marker_position != std::string::npos, "npy header has no key " << key);
    const auto value_start = marker_position + marker.size();
    const auto value_end = header.find(terminator, value_start + 1);
    KAEL_CHECK(value_end != std::string::npos, "npy header has a malformed value for " << key);
    return header.substr(value_start, value_end - value_start + 1);
}

DType parse_dtype(const std::string& descr) {
    if (descr == "'<f4'") {
        return DType::F32;
    }
    if (descr == "'<f8'") {
        return DType::F64;
    }
    if (descr == "'<i8'") {
        return DType::I64;
    }
    if (descr == "'|b1'") {
        return DType::Bool;
    }
    throw Error("unsupported npy dtype " + descr);
}

Shape parse_shape(const std::string& tuple) {
    Shape shape;
    std::istringstream stream(tuple.substr(1, tuple.size() - 2));
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.find_first_not_of(' ') != std::string::npos) {
            shape.push_back(std::stoll(item));
        }
    }
    return shape;
}

std::string read_header(std::ifstream& file) {
    std::array<char, 6> magic{};
    file.read(magic.data(), magic.size());
    KAEL_CHECK(magic == kMagic, "not an npy file");

    std::array<unsigned char, 4> version_and_length{};
    file.read(reinterpret_cast<char*>(version_and_length.data()), version_and_length.size());
    KAEL_CHECK(version_and_length[0] == 1, "only npy version 1 is supported");

    const std::size_t length = version_and_length[2] + (std::size_t{version_and_length[3]} << 8);
    std::string header(length, ' ');
    file.read(header.data(), static_cast<std::streamsize>(length));
    return header;
}

} // namespace

Tensor read_npy(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    KAEL_CHECK(file.is_open(), "cannot open npy file " << path);

    const std::string header = read_header(file);
    KAEL_CHECK(header_value(header, "fortran_order", 'e') == "False",
               "fortran order npy files are not supported: " << path);

    Tensor tensor = empty(parse_shape(header_value(header, "shape", ')')),
                          parse_dtype(header_value(header, "descr", '\'')));
    file.read(reinterpret_cast<char*>(tensor.data_ptr_raw()),
              static_cast<std::streamsize>(tensor.nbytes()));
    KAEL_CHECK(file.good() || tensor.nbytes() == 0, "truncated npy file " << path);
    return tensor;
}

} // namespace kael::test
