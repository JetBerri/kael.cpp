#include "kael/data/mnist.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "kael/factory.h"
#include "kael/ops.h"

namespace kael::data {

namespace {

constexpr std::uint32_t kImagesMagic = 0x00000803;
constexpr std::uint32_t kLabelsMagic = 0x00000801;
constexpr double kMnistMean = 0.1307;
constexpr double kMnistStd = 0.3081;

std::uint32_t read_big_endian(std::ifstream& file, const std::filesystem::path& path) {
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    KAEL_CHECK(file.good(), "IDX file " << path << " is truncated in its header");
    return (std::uint32_t{bytes[0]} << 24) | (std::uint32_t{bytes[1]} << 16) |
           (std::uint32_t{bytes[2]} << 8) | std::uint32_t{bytes[3]};
}

std::vector<std::int64_t> read_header(std::ifstream& file, const std::filesystem::path& path,
                                      std::uint32_t magic, std::size_t dimensions) {
    KAEL_CHECK(file.is_open(), "cannot open IDX file " << path);
    const std::uint32_t found = read_big_endian(file, path);
    KAEL_CHECK(found == magic, "IDX file " << path << " has magic number " << found << ", expected "
                                           << magic);
    std::vector<std::int64_t> sizes;
    for (std::size_t i = 0; i < dimensions; ++i) {
        sizes.push_back(static_cast<std::int64_t>(read_big_endian(file, path)));
    }
    return sizes;
}

std::vector<unsigned char> read_payload(std::ifstream& file, const std::filesystem::path& path,
                                        const Shape& shape) {
    const std::int64_t expected = numel(shape);
    const std::streampos start = file.tellg();
    file.seekg(0, std::ios::end);
    const std::int64_t available = file.tellg() - start;
    KAEL_CHECK(available == expected,
               "IDX file " << path << " has " << available << " data bytes, expected " << expected);
    file.seekg(start);
    std::vector<unsigned char> payload(static_cast<std::size_t>(expected));
    file.read(reinterpret_cast<char*>(payload.data()), expected);
    KAEL_CHECK(file.good() || expected == 0, "cannot read IDX data from " << path);
    return payload;
}

} // namespace

Tensor read_idx_images(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<std::int64_t> sizes = read_header(file, path, kImagesMagic, 3);
    const Shape shape{sizes[0], 1, sizes[1], sizes[2]};
    const std::vector<unsigned char> payload = read_payload(file, path, shape);

    Tensor images = empty(shape, DType::F32);
    float* data = images.data_ptr<float>();
    for (std::size_t i = 0; i < payload.size(); ++i) {
        data[i] = static_cast<float>(payload[i]) / 255.0f;
    }
    return images;
}

Tensor read_idx_labels(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<std::int64_t> sizes = read_header(file, path, kLabelsMagic, 1);
    const Shape shape{sizes[0]};
    const std::vector<unsigned char> payload = read_payload(file, path, shape);

    Tensor labels = empty(shape, DType::I64);
    std::int64_t* data = labels.data_ptr<std::int64_t>();
    for (std::size_t i = 0; i < payload.size(); ++i) {
        data[i] = payload[i];
    }
    return labels;
}

MnistSplit load_mnist(const std::filesystem::path& directory, MnistPart part) {
    const std::string prefix = part == MnistPart::Train ? "train" : "t10k";
    const Tensor images = read_idx_images(directory / (prefix + "-images-idx3-ubyte"));
    const Tensor labels = read_idx_labels(directory / (prefix + "-labels-idx1-ubyte"));
    KAEL_CHECK(images.size(0) == labels.size(0), "MNIST has " << images.size(0) << " images but "
                                                              << labels.size(0) << " labels");
    return {(images - kMnistMean) / kMnistStd, labels};
}

} // namespace kael::data
