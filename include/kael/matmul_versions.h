#pragma once

#include <cstdint>

namespace kael::matmul_versions {

struct Sizes {
    std::int64_t rows;
    std::int64_t inner;
    std::int64_t columns;
};

template <typename T> void v1_ijk(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v2_ikj(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v3_tiled(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v4_packed(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v5_simd(const T* a, const T* b, T* out, const Sizes& sizes);
template <typename T> void v6_threaded(const T* a, const T* b, T* out, const Sizes& sizes);

[[nodiscard]] bool blas_available() noexcept;
template <typename T> void blas(const T* a, const T* b, T* out, const Sizes& sizes);

template <typename T> void multiply(const T* a, const T* b, T* out, const Sizes& sizes);

} // namespace kael::matmul_versions
