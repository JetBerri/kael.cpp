#include "kael/matmul_versions.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(KAEL_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(KAEL_USE_CBLAS)
#include <cblas.h>
#endif

#include "kael/error.h"
#include "kael/parallel.h"

namespace kael::matmul_versions {

namespace {

using BlockSizes = Sizes;

constexpr BlockSizes kTileBlocks{64, 256, 2048};
constexpr BlockSizes kPackedBlocks{384, 512, 4096};
constexpr std::int64_t kSmallProblem = std::int64_t{32} * 32 * 32;
constexpr std::int64_t kThreadedProblem = std::int64_t{128} * 128 * 128;
constexpr std::size_t kPackAlignment = 64;

struct FreeAligned {
    void operator()(void* memory) const noexcept {
        ::operator delete(memory, std::align_val_t{kPackAlignment});
    }
};

template <typename T> using PackBuffer = std::unique_ptr<T[], FreeAligned>;

template <typename T> PackBuffer<T> allocate_pack(std::int64_t count) {
    return PackBuffer<T>(static_cast<T*>(::operator new(static_cast<std::size_t>(count) * sizeof(T),
                                                        std::align_val_t{kPackAlignment})));
}

template <typename T, std::int64_t Rows, std::int64_t Columns> struct GenericKernel {
    static constexpr std::int64_t kRows = Rows;
    static constexpr std::int64_t kColumns = Columns;

    static void run(const T* a, const T* b, std::int64_t inner, T* result) {
        std::array<T, static_cast<std::size_t>(Rows * Columns)> accumulators{};
        for (std::int64_t k = 0; k < inner; ++k) {
            for (std::int64_t i = 0; i < Rows; ++i) {
                const T a_value = a[i];
                T* row = accumulators.data() + i * Columns;
                for (std::int64_t j = 0; j < Columns; ++j) {
                    row[j] += a_value * b[j];
                }
            }
            a += Rows;
            b += Columns;
        }
        std::copy(accumulators.begin(), accumulators.end(), result);
    }
};

#if defined(__ARM_NEON)

struct NeonFloatKernel {
    static constexpr std::int64_t kRows = 8;
    static constexpr std::int64_t kColumns = 12;

    static void run(const float* a, const float* b, std::int64_t inner, float* result) {
        float32x4_t c[12][2];
        for (auto& column : c) {
            column[0] = vdupq_n_f32(0.0f);
            column[1] = vdupq_n_f32(0.0f);
        }
        for (std::int64_t k = 0; k < inner; ++k) {
            const float32x4_t a_low = vld1q_f32(a);
            const float32x4_t a_high = vld1q_f32(a + 4);
            const float32x4_t b0 = vld1q_f32(b);
            const float32x4_t b1 = vld1q_f32(b + 4);
            const float32x4_t b2 = vld1q_f32(b + 8);
#define KAEL_FMA_COLUMN(column, source, lane)                                                      \
    c[column][0] = vfmaq_laneq_f32(c[column][0], a_low, source, lane);                             \
    c[column][1] = vfmaq_laneq_f32(c[column][1], a_high, source, lane)
            KAEL_FMA_COLUMN(0, b0, 0);
            KAEL_FMA_COLUMN(1, b0, 1);
            KAEL_FMA_COLUMN(2, b0, 2);
            KAEL_FMA_COLUMN(3, b0, 3);
            KAEL_FMA_COLUMN(4, b1, 0);
            KAEL_FMA_COLUMN(5, b1, 1);
            KAEL_FMA_COLUMN(6, b1, 2);
            KAEL_FMA_COLUMN(7, b1, 3);
            KAEL_FMA_COLUMN(8, b2, 0);
            KAEL_FMA_COLUMN(9, b2, 1);
            KAEL_FMA_COLUMN(10, b2, 2);
            KAEL_FMA_COLUMN(11, b2, 3);
#undef KAEL_FMA_COLUMN
            a += kRows;
            b += kColumns;
        }
        float by_column[8];
        for (std::int64_t j = 0; j < kColumns; ++j) {
            vst1q_f32(by_column, c[j][0]);
            vst1q_f32(by_column + 4, c[j][1]);
            for (std::int64_t i = 0; i < kRows; ++i) {
                result[i * kColumns + j] = by_column[i];
            }
        }
    }
};

struct NeonDoubleKernel {
    static constexpr std::int64_t kRows = 6;
    static constexpr std::int64_t kColumns = 8;

    static void run(const double* a, const double* b, std::int64_t inner, double* result) {
        float64x2_t c[8][3];
        for (auto& column : c) {
            column[0] = vdupq_n_f64(0.0);
            column[1] = vdupq_n_f64(0.0);
            column[2] = vdupq_n_f64(0.0);
        }
        for (std::int64_t k = 0; k < inner; ++k) {
            const float64x2_t a0 = vld1q_f64(a);
            const float64x2_t a1 = vld1q_f64(a + 2);
            const float64x2_t a2 = vld1q_f64(a + 4);
            for (std::int64_t j = 0; j < kColumns; ++j) {
                const float64x2_t b_value = vdupq_n_f64(b[j]);
                c[j][0] = vfmaq_f64(c[j][0], a0, b_value);
                c[j][1] = vfmaq_f64(c[j][1], a1, b_value);
                c[j][2] = vfmaq_f64(c[j][2], a2, b_value);
            }
            a += kRows;
            b += kColumns;
        }
        double by_column[6];
        for (std::int64_t j = 0; j < kColumns; ++j) {
            vst1q_f64(by_column, c[j][0]);
            vst1q_f64(by_column + 2, c[j][1]);
            vst1q_f64(by_column + 4, c[j][2]);
            for (std::int64_t i = 0; i < kRows; ++i) {
                result[i * kColumns + j] = by_column[i];
            }
        }
    }
};

template <typename T> struct SimdKernelFor;
template <> struct SimdKernelFor<float> {
    using Type = NeonFloatKernel;
};
template <> struct SimdKernelFor<double> {
    using Type = NeonDoubleKernel;
};

#else

template <typename T> struct SimdKernelFor {
    using Type = GenericKernel<T, 6, 8>;
};

#endif

template <typename T, typename Kernel>
void pack_rows(const T* a, std::int64_t stride, std::int64_t rows, std::int64_t inner, T* packed) {
    for (std::int64_t top = 0; top < rows; top += Kernel::kRows) {
        const std::int64_t height = std::min(Kernel::kRows, rows - top);
        for (std::int64_t k = 0; k < inner; ++k) {
            for (std::int64_t i = 0; i < height; ++i) {
                packed[i] = a[(top + i) * stride + k];
            }
            std::fill(packed + height, packed + Kernel::kRows, T{0});
            packed += Kernel::kRows;
        }
    }
}

template <typename T, typename Kernel>
void pack_column_panels(const T* b, std::int64_t stride, std::int64_t inner, std::int64_t columns,
                        std::int64_t first_panel, std::int64_t last_panel, T* packed) {
    T* cursor = packed + first_panel * inner * Kernel::kColumns;
    for (std::int64_t panel = first_panel; panel < last_panel; ++panel) {
        const std::int64_t left = panel * Kernel::kColumns;
        const std::int64_t width = std::min(Kernel::kColumns, columns - left);
        for (std::int64_t k = 0; k < inner; ++k) {
            const T* row = b + k * stride + left;
            std::copy(row, row + width, cursor);
            std::fill(cursor + width, cursor + Kernel::kColumns, T{0});
            cursor += Kernel::kColumns;
        }
    }
}

template <typename T, typename Kernel>
void multiply_row_block(const T* a, std::int64_t a_stride, const T* packed_b, T* out,
                        std::int64_t out_stride, std::int64_t rows, std::int64_t inner,
                        std::int64_t columns, T* packed_a) {
    pack_rows<T, Kernel>(a, a_stride, rows, inner, packed_a);
    std::array<T, static_cast<std::size_t>(Kernel::kRows * Kernel::kColumns)> result;
    for (std::int64_t left = 0; left < columns; left += Kernel::kColumns) {
        const std::int64_t width = std::min(Kernel::kColumns, columns - left);
        const T* b_panel = packed_b + (left / Kernel::kColumns) * inner * Kernel::kColumns;
        for (std::int64_t top = 0; top < rows; top += Kernel::kRows) {
            const std::int64_t height = std::min(Kernel::kRows, rows - top);
            const T* a_panel = packed_a + (top / Kernel::kRows) * inner * Kernel::kRows;
            Kernel::run(a_panel, b_panel, inner, result.data());
            for (std::int64_t i = 0; i < height; ++i) {
                T* out_row = out + (top + i) * out_stride + left;
                const T* result_row = result.data() + i * Kernel::kColumns;
                for (std::int64_t j = 0; j < width; ++j) {
                    out_row[j] += result_row[j];
                }
            }
        }
    }
}

template <std::int64_t Unit> std::int64_t rounded_up(std::int64_t value) {
    return (value + Unit - 1) / Unit * Unit;
}

template <typename Kernel, bool Threaded>
std::int64_t row_block_for(std::int64_t rows, std::int64_t largest) {
    if constexpr (Threaded) {
        const std::int64_t threads = get_num_threads();
        return std::min(largest, rounded_up<Kernel::kRows>((rows + threads - 1) / threads));
    } else {
        return largest;
    }
}

template <bool Threaded, typename Body>
void run_blocks(std::int64_t count, std::int64_t grain, Body body) {
    if constexpr (Threaded) {
        parallel_for(0, count, grain, body);
    } else {
        body(0, count);
    }
}

template <typename T, typename Kernel, bool Threaded>
void multiply_packed(const T* a, const T* b, T* out, const Sizes& sizes) {
    const auto [rows, inner, columns] = sizes;
    if (rows == 0 || inner == 0 || columns == 0) {
        return;
    }
    const BlockSizes blocks{rounded_up<Kernel::kRows>(kPackedBlocks.rows), kPackedBlocks.inner,
                            rounded_up<Kernel::kColumns>(kPackedBlocks.columns)};
    const std::int64_t row_block = row_block_for<Kernel, Threaded>(rows, blocks.rows);
    const std::int64_t column_block = std::min(blocks.columns, columns);
    const std::int64_t inner_block = std::min(blocks.inner, inner);
    const std::int64_t packed_b_size = rounded_up<Kernel::kColumns>(column_block) * inner_block;
    const std::int64_t packed_a_size = rounded_up<Kernel::kRows>(std::min(row_block, rows)) * inner_block;
    const PackBuffer<T> packed_b = allocate_pack<T>(packed_b_size);

    for (std::int64_t jc = 0; jc < columns; jc += blocks.columns) {
        const std::int64_t nc = std::min(blocks.columns, columns - jc);
        for (std::int64_t pc = 0; pc < inner; pc += blocks.inner) {
            const std::int64_t kc = std::min(blocks.inner, inner - pc);
            const bool worthwhile = rows * kc * nc >= kThreadedProblem;

            const std::int64_t panel_count = rounded_up<Kernel::kColumns>(nc) / Kernel::kColumns;
            const std::int64_t panel_elements = kc * Kernel::kColumns;
            run_blocks<Threaded>(panel_count, std::max<std::int64_t>(1, kElementGrain / panel_elements),
                                 [&](std::int64_t first, std::int64_t last) {
                                     pack_column_panels<T, Kernel>(b + pc * columns + jc, columns, kc,
                                                                   nc, first, last, packed_b.get());
                                 });

            const std::int64_t block_count = (rows + row_block - 1) / row_block;
            run_blocks<Threaded>(block_count, worthwhile ? 1 : block_count,
                                 [&](std::int64_t first, std::int64_t last) {
                                     const PackBuffer<T> packed_a = allocate_pack<T>(packed_a_size);
                                     for (std::int64_t block = first; block < last; ++block) {
                                         const std::int64_t ic = block * row_block;
                                         const std::int64_t mc = std::min(row_block, rows - ic);
                                         multiply_row_block<T, Kernel>(
                                             a + ic * inner + pc, inner, packed_b.get(),
                                             out + ic * columns + jc, columns, mc, kc, nc,
                                             packed_a.get());
                                     }
                                 });
        }
    }
}

template <typename T> using SimdKernel = typename SimdKernelFor<T>::Type;
template <typename T> using ScalarKernel = GenericKernel<T, 4, 4>;

} // namespace

template <typename T> void v1_ijk(const T* a, const T* b, T* out, const Sizes& sizes) {
    const auto [rows, inner, columns] = sizes;
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < columns; ++j) {
            T total{0};
            for (std::int64_t k = 0; k < inner; ++k) {
                total += a[i * inner + k] * b[k * columns + j];
            }
            out[i * columns + j] += total;
        }
    }
}

template <typename T> void v2_ikj(const T* a, const T* b, T* out, const Sizes& sizes) {
    const auto [rows, inner, columns] = sizes;
    for (std::int64_t i = 0; i < rows; ++i) {
        T* out_row = out + i * columns;
        for (std::int64_t k = 0; k < inner; ++k) {
            const T a_ik = a[i * inner + k];
            const T* b_row = b + k * columns;
            for (std::int64_t j = 0; j < columns; ++j) {
                out_row[j] += a_ik * b_row[j];
            }
        }
    }
}

template <typename T> void v3_tiled(const T* a, const T* b, T* out, const Sizes& sizes) {
    const auto [rows, inner, columns] = sizes;
    for (std::int64_t jc = 0; jc < columns; jc += kTileBlocks.columns) {
        const std::int64_t j_end = std::min(columns, jc + kTileBlocks.columns);
        for (std::int64_t pc = 0; pc < inner; pc += kTileBlocks.inner) {
            const std::int64_t k_end = std::min(inner, pc + kTileBlocks.inner);
            for (std::int64_t ic = 0; ic < rows; ic += kTileBlocks.rows) {
                const std::int64_t i_end = std::min(rows, ic + kTileBlocks.rows);
                for (std::int64_t i = ic; i < i_end; ++i) {
                    T* out_row = out + i * columns;
                    for (std::int64_t k = pc; k < k_end; ++k) {
                        const T a_ik = a[i * inner + k];
                        const T* b_row = b + k * columns;
                        for (std::int64_t j = jc; j < j_end; ++j) {
                            out_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T> void v4_packed(const T* a, const T* b, T* out, const Sizes& sizes) {
    multiply_packed<T, ScalarKernel<T>, false>(a, b, out, sizes);
}

template <typename T> void v5_simd(const T* a, const T* b, T* out, const Sizes& sizes) {
    multiply_packed<T, SimdKernel<T>, false>(a, b, out, sizes);
}

template <typename T> void v6_threaded(const T* a, const T* b, T* out, const Sizes& sizes) {
    multiply_packed<T, SimdKernel<T>, true>(a, b, out, sizes);
}

bool blas_available() noexcept {
#if defined(KAEL_USE_ACCELERATE) || defined(KAEL_USE_CBLAS)
    return true;
#else
    return false;
#endif
}

template <typename T>
void blas([[maybe_unused]] const T* a, [[maybe_unused]] const T* b, [[maybe_unused]] T* out,
          [[maybe_unused]] const Sizes& sizes) {
#if defined(KAEL_USE_ACCELERATE) || defined(KAEL_USE_CBLAS)
    if (sizes.rows == 0 || sizes.inner == 0 || sizes.columns == 0) {
        return;
    }
    const auto rows = static_cast<int>(sizes.rows);
    const auto inner = static_cast<int>(sizes.inner);
    const auto columns = static_cast<int>(sizes.columns);
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inner, 1.0f, a, inner,
                    b, columns, 1.0f, out, columns);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inner, 1.0, a, inner,
                    b, columns, 1.0, out, columns);
    }
#else
    KAEL_CHECK(false, "kael was built without BLAS; configure with KAEL_USE_BLAS=ON");
#endif
}

template <typename T> void multiply(const T* a, const T* b, T* out, const Sizes& sizes) {
    if (blas_available()) {
        blas(a, b, out, sizes);
    } else if (sizes.rows * sizes.inner * sizes.columns <= kSmallProblem) {
        v2_ikj(a, b, out, sizes);
    } else {
        v6_threaded(a, b, out, sizes);
    }
}

#define KAEL_INSTANTIATE_MATMUL(T)                                                                 \
    template void v1_ijk<T>(const T*, const T*, T*, const Sizes&);                                 \
    template void v2_ikj<T>(const T*, const T*, T*, const Sizes&);                                 \
    template void v3_tiled<T>(const T*, const T*, T*, const Sizes&);                               \
    template void v4_packed<T>(const T*, const T*, T*, const Sizes&);                              \
    template void v5_simd<T>(const T*, const T*, T*, const Sizes&);                                \
    template void v6_threaded<T>(const T*, const T*, T*, const Sizes&);                            \
    template void blas<T>(const T*, const T*, T*, const Sizes&);                                   \
    template void multiply<T>(const T*, const T*, T*, const Sizes&);

KAEL_INSTANTIATE_MATMUL(float)
KAEL_INSTANTIATE_MATMUL(double)

#undef KAEL_INSTANTIATE_MATMUL

} // namespace kael::matmul_versions
