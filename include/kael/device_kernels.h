#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kael/dtype.h"

namespace kael::device {

inline constexpr int kMaxDims = 8;
inline constexpr int kMaxOperands = 4;

enum class ElementOp {
    Plus,
    Minus,
    Times,
    Divides,
    Power,
    Maximum,
    Minimum,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    IsZero,
    Negate,
    Absolute,
    Sign,
    Exp,
    Log,
    Log1p,
    Expm1,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    GeluExact,
    GeluTanh,
    Clamp,
    Where,
    MaskedFill,
};

enum class ReduceOp { Sum, Max, Min };

// Integer arguments travel apart from real ones: a double cannot hold every int64 exactly.
struct ElementParameters {
    std::array<double, 2> reals{};
    std::array<std::int64_t, 2> integers{};

    template <typename T> [[nodiscard]] T get(std::size_t index) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(integers[index]);
        } else {
            return static_cast<T>(reals[index]);
        }
    }
};

struct ElementLayout {
    int dims;
    int operands;
    std::int64_t count;
    std::int64_t shape[kMaxDims];
    std::int64_t strides[kMaxOperands][kMaxDims];
    std::byte* base[kMaxOperands];
};

struct IndexLayout {
    int dims;
    int axis;
    std::int64_t count;
    std::int64_t shape[kMaxDims];
    std::int64_t data_strides[kMaxDims];
    std::int64_t index_strides[kMaxDims];
    std::int64_t source_strides[kMaxDims];
};

struct MatmulSizes {
    std::int64_t batches;
    std::int64_t rows;
    std::int64_t inner;
    std::int64_t columns;
};

[[nodiscard]] bool available();
void synchronize();

[[nodiscard]] std::byte* allocate(std::int64_t bytes);
void release(std::byte* block) noexcept;
void copy_to_device(std::byte* target, const std::byte* source, std::int64_t bytes);
void copy_to_host(std::byte* target, const std::byte* source, std::int64_t bytes);
void copy_within_device(std::byte* target, const std::byte* source, std::int64_t bytes);

void fill(DType dtype, std::byte* target, std::int64_t count, const void* value);
void convert(DType out, DType in, const ElementLayout& layout);
void elementwise(ElementOp op, DType out, DType in, const ElementLayout& layout,
                 ElementParameters parameters);
void reduce(ReduceOp op, DType out, DType in, std::byte* results, const std::byte* data,
            std::int64_t chunk_size, std::int64_t output_count);
void matmul(DType dtype, const std::byte* a, const std::byte* b, std::byte* out,
            const MatmulSizes& sizes);
void gather(DType dtype, std::byte* out, const std::byte* data, const std::int64_t* indices,
            const IndexLayout& layout);
void scatter_add(DType dtype, std::byte* data, const std::int64_t* indices, const std::byte* source,
                 const IndexLayout& layout);

} // namespace kael::device
