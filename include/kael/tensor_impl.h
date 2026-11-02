#pragma once

#include <cstdint>
#include <memory>

#include "kael/dtype.h"
#include "kael/shape.h"
#include "kael/storage.h"

namespace kael {

// The metadata that interprets a Storage as one concrete tensor.
//
// Several TensorImpls can point to the same Storage: that is what a view is.
// Strides and offset are in ELEMENTS, not bytes; bytes only appear when
// computing a pointer (offset * element_size).
struct TensorImpl {
    std::shared_ptr<Storage> storage;
    Shape shape;
    Strides strides;
    std::int64_t offset = 0; // first element of the tensor inside the storage
    DType dtype = DType::F32;
    Device device = Device::CPU; // must match storage->device()

    // Autograd fields arrive in phase 7:
    //   bool requires_grad; std::shared_ptr<TensorImpl> grad; std::shared_ptr<Node> grad_fn;

    [[nodiscard]] std::int64_t numel() const { return kael::numel(shape); }
    [[nodiscard]] std::int64_t dim() const { return static_cast<std::int64_t>(shape.size()); }

    // True if the elements are laid out in row-major order without gaps.
    // Strides of size-1 dimensions are ignored: you never step along them.
    [[nodiscard]] bool is_contiguous() const;

    // Checks that the metadata is consistent and that every element it can
    // address lies inside the storage. Throws kael::Error otherwise.
    // Called whenever an impl is created (factories, views).
    void validate() const;
};

} // namespace kael
