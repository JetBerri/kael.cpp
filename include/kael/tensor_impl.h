#pragma once

#include <cstdint>
#include <memory>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/storage.h"

namespace kael {

// Strides and offset count elements, not bytes.
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;
    std::int64_t offset = 0;
    DType dtype = DType::F32;
    Device device = Device::CPU;

    [[nodiscard]] std::int64_t numel() const { return kael::numel(shape); }
    [[nodiscard]] std::int64_t dim() const { return static_cast<std::int64_t>(shape.size()); }
    [[nodiscard]] bool is_contiguous() const;
    void validate() const;
};

} // namespace kael
