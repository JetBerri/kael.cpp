#include "kael/io/safetensors.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <system_error>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "kael/factory.h"
#include "kael/storage.h"
#include "kael/tensor_impl.h"

#include "internal/checked_math.h"

namespace kael::io {

namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;
constexpr std::size_t kHeaderAlignment = 8;
constexpr const char* kMetadataKey = "__metadata__";

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return "F32";
    case DType::F64:
        return "F64";
    case DType::I64:
        return "I64";
    case DType::Bool:
        return "BOOL";
    }
    detail::unreachable();
}

DType dtype_from_name(const std::string& name) {
    for (const DType dtype : {DType::F32, DType::F64, DType::I64, DType::Bool}) {
        if (name == dtype_name(dtype)) {
            return dtype;
        }
    }
    throw TypeError("safetensors dtype '" + name +
                    "' is not supported: kael has F32, F64, I64 and BOOL");
}

std::int64_t non_negative_integer(const Json& value, const std::string& what) {
    KAEL_CHECK_VALUE(value.is_number_integer(), what << " must be an integer");
    if (value.is_number_unsigned()) {
        const auto unsigned_value = value.get<std::uint64_t>();
        KAEL_CHECK_VALUE(unsigned_value <=
                             static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                         what << " does not fit in int64");
        return static_cast<std::int64_t>(unsigned_value);
    }
    const auto signed_value = value.get<std::int64_t>();
    KAEL_CHECK_VALUE(signed_value >= 0, what << " must not be negative");
    return signed_value;
}

struct Entry {
    std::string name;
    DType dtype;
    Shape shape;
    std::int64_t begin;
    std::int64_t end;
};

Entry parse_entry(const std::string& name, const Json& description) {
    KAEL_CHECK_VALUE(description.is_object(), "safetensors entry " << name << " is not an object");
    KAEL_CHECK_VALUE(description.contains("dtype") && description.at("dtype").is_string(),
                     "safetensors entry " << name << " has no dtype");
    KAEL_CHECK_VALUE(description.contains("shape") && description.at("shape").is_array(),
                     "safetensors entry " << name << " has no shape");
    KAEL_CHECK_VALUE(description.contains("data_offsets") &&
                         description.at("data_offsets").is_array() &&
                         description.at("data_offsets").size() == 2,
                     "safetensors entry " << name << " needs two data offsets");

    Entry entry{name, dtype_from_name(description.at("dtype").get<std::string>()), {}, 0, 0};
    for (const Json& size : description.at("shape")) {
        entry.shape.push_back(non_negative_integer(size, "a dimension of " + name));
    }
    const Json& offsets = description.at("data_offsets");
    entry.begin = non_negative_integer(offsets[0], "an offset of " + name);
    entry.end = non_negative_integer(offsets[1], "an offset of " + name);
    KAEL_CHECK_VALUE(entry.begin <= entry.end,
                     "safetensors entry " << name << " ends before it starts");
    const std::int64_t expected =
        detail::checked_mul(numel(entry.shape), element_size(entry.dtype), "tensor size in bytes");
    KAEL_CHECK_VALUE(entry.end - entry.begin == expected,
                     "safetensors entry " << name << " has " << entry.end - entry.begin
                                          << " bytes but its shape and dtype need " << expected);
    return entry;
}

void check_layout(std::vector<Entry>& entries, std::int64_t data_size) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    std::int64_t position = 0;
    for (const Entry& entry : entries) {
        KAEL_CHECK_VALUE(entry.begin == position,
                         "safetensors entry " << entry.name << " starts at " << entry.begin
                                              << " but the previous one ended at " << position);
        position = entry.end;
    }
    KAEL_CHECK_VALUE(position == data_size, "safetensors tensors cover "
                                                << position << " bytes but the file has "
                                                << data_size);
}

void check_bools(const Tensor& t, const std::string& name) {
    const auto* values = t.data_ptr<std::uint8_t>();
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        KAEL_CHECK_VALUE(values[i] <= 1, "safetensors bool tensor " << name << " has the byte "
                                                                    << static_cast<int>(values[i]));
    }
}

std::string dump_header(const Json& header) {
    try {
        return header.dump();
    } catch (const Json::type_error& error) {
        throw ValueError(std::string("tensor names and metadata must be valid UTF-8: ") +
                         error.what());
    }
}

// Without fsync a crash after the rename can leave a renamed but empty or partial file.
void flush_to_disk(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    KAEL_CHECK(descriptor >= 0, "cannot open " << path << " to flush it to disk");
    const int result = ::fsync(descriptor);
    ::close(descriptor);
    KAEL_CHECK(result == 0, "cannot flush " << path << " to disk");
}

struct TemporaryFileRemover {
    std::filesystem::path path;

    ~TemporaryFileRemover() {
        std::error_code ignored;
        if (!path.empty()) {
            std::filesystem::remove(path, ignored);
        }
    }
};

std::uint64_t next_temporary_suffix() {
    static std::atomic<std::uint64_t> counter{0};
    const auto clock = std::chrono::steady_clock::now().time_since_epoch().count();
    return static_cast<std::uint64_t>(clock) + counter.fetch_add(1);
}

// Private pages: writing to a mapped tensor copies the page and never reaches the file.
std::shared_ptr<void> map_file(const std::filesystem::path& path, std::uint64_t expected_size) {
#if defined(_WIN32)
    (void)path;
    (void)expected_size;
    return nullptr;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    KAEL_CHECK(descriptor >= 0, "cannot open safetensors file " << path);
    struct stat status{};
    const bool same_size = ::fstat(descriptor, &status) == 0 &&
                           static_cast<std::uint64_t>(status.st_size) == expected_size;
    void* address = same_size ? ::mmap(nullptr, static_cast<std::size_t>(expected_size),
                                       PROT_READ | PROT_WRITE, MAP_PRIVATE, descriptor, 0)
                              : MAP_FAILED;
    ::close(descriptor);
    KAEL_CHECK(same_size, "safetensors file " << path << " changed while it was being read");
    if (address == MAP_FAILED) {
        return nullptr;
    }
    return {address, [expected_size](void* mapped) {
                ::munmap(mapped, static_cast<std::size_t>(expected_size));
            }};
#endif
}

Tensor tensor_over_mapping(const std::shared_ptr<void>& mapping, std::uint64_t position,
                           const Entry& entry) {
    auto impl = std::make_shared<TensorImpl>();
    auto* first = static_cast<std::byte*>(mapping.get()) + position;
    impl->storage =
        std::make_shared<Storage>(Storage::external(first, entry.end - entry.begin, mapping));
    impl->shape = entry.shape;
    impl->strides = contiguous_strides(entry.shape);
    impl->dtype = entry.dtype;
    return Tensor::from_impl(std::move(impl));
}

} // namespace

void save_safetensors(const std::filesystem::path& path, const TensorMap& tensors,
                      const Metadata& metadata) {
    Json header = Json::object();
    if (!metadata.empty()) {
        header[kMetadataKey] = metadata;
    }
    std::int64_t position = 0;
    std::vector<Tensor> payloads;
    payloads.reserve(tensors.size());
    for (const auto& [name, tensor] : tensors) {
        KAEL_CHECK_VALUE(name != kMetadataKey, "a tensor cannot be named " << kMetadataKey);
        payloads.push_back(tensor.detach().cpu().contiguous());
        const std::int64_t end = position + tensor.nbytes();
        header[name] = {{"dtype", dtype_name(tensor.dtype())},
                        {"shape", tensor.shape()},
                        {"data_offsets", {position, end}}};
        position = end;
    }

    std::string text = dump_header(header);
    text.append((kHeaderAlignment - (text.size() % kHeaderAlignment)) % kHeaderAlignment, ' ');
    const std::uint64_t header_size = text.size();

    // Mapped tensors may read from the file being replaced: truncating it would invalidate them.
    std::filesystem::path temporary = path;
    temporary +=
        ".tmp" + std::to_string(::getpid()) + "." + std::to_string(next_temporary_suffix());
    TemporaryFileRemover remover{temporary};
    std::ofstream file(temporary, std::ios::binary);
    KAEL_CHECK(file.is_open(), "cannot write safetensors file " << temporary);
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const Tensor& payload : payloads) {
        if (payload.nbytes() > 0) {
            file.write(reinterpret_cast<const char*>(payload.data_ptr_raw()),
                       static_cast<std::streamsize>(payload.nbytes()));
        }
    }
    file.close();
    KAEL_CHECK(file.good(), "failed writing safetensors file " << temporary);
    flush_to_disk(temporary);
    std::filesystem::rename(temporary, path);
    remover.path.clear();
    flush_to_disk(path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path());
}

SafetensorsFile read_safetensors(const std::filesystem::path& path, LoadMode mode) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    KAEL_CHECK(file.is_open(), "cannot open safetensors file " << path);
    const auto file_size = static_cast<std::uint64_t>(file.tellg());
    KAEL_CHECK_VALUE(file_size >= sizeof(std::uint64_t),
                     "safetensors file " << path << " is too short");
    file.seekg(0);

    std::uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    KAEL_CHECK_VALUE(header_size <= kMaxHeaderBytes &&
                         header_size <= file_size - sizeof(header_size),
                     "safetensors header of " << header_size << " bytes does not fit the file");

    std::string text(static_cast<std::size_t>(header_size), '\0');
    file.read(text.data(), static_cast<std::streamsize>(header_size));
    KAEL_CHECK_VALUE(file.good(), "safetensors file " << path << " is truncated");

    const Json header = Json::parse(text, nullptr, false);
    KAEL_CHECK_VALUE(!header.is_discarded() && header.is_object(),
                     "safetensors header of " << path << " is not a JSON object");

    SafetensorsFile result;
    std::vector<Entry> entries;
    for (const auto& [name, value] : header.items()) {
        if (name == kMetadataKey) {
            KAEL_CHECK_VALUE(value.is_object(), "safetensors metadata is not an object");
            for (const auto& [key, text_value] : value.items()) {
                KAEL_CHECK_VALUE(text_value.is_string(),
                                 "safetensors metadata " << key << " is not a string");
                result.metadata.emplace(key, text_value.get<std::string>());
            }
        } else {
            entries.push_back(parse_entry(name, value));
        }
    }

    const std::uint64_t data_size = file_size - sizeof(header_size) - header_size;
    check_layout(entries, static_cast<std::int64_t>(data_size));
    const auto data_start = static_cast<std::streamoff>(sizeof(header_size) + header_size);
    const std::shared_ptr<void> mapping =
        mode == LoadMode::Mapped ? map_file(path, file_size) : nullptr;
    for (const Entry& entry : entries) {
        const auto position = static_cast<std::uint64_t>(data_start + entry.begin);
        const auto alignment = static_cast<std::uint64_t>(element_size(entry.dtype));
        const bool over_mapping = mapping && entry.end > entry.begin && position % alignment == 0;
        Tensor tensor = over_mapping ? tensor_over_mapping(mapping, position, entry)
                                     : empty(entry.shape, entry.dtype);
        if (!over_mapping && tensor.nbytes() > 0) {
            file.seekg(data_start + entry.begin);
            file.read(reinterpret_cast<char*>(tensor.data_ptr_raw()),
                      static_cast<std::streamsize>(tensor.nbytes()));
            KAEL_CHECK_VALUE(file.good(), "safetensors file " << path << " is truncated");
        }
        if (entry.dtype == DType::Bool) {
            check_bools(tensor, entry.name);
        }
        result.tensors.emplace(entry.name, std::move(tensor));
    }
    return result;
}

TensorMap load_safetensors(const std::filesystem::path& path, LoadMode mode) {
    return read_safetensors(path, mode).tensors;
}

} // namespace kael::io
