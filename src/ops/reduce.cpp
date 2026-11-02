#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <type_traits>

#include "internal/nan_aware.h"
#include "kael/factory.h"
#include "kael/ops.h"

namespace kael {

namespace {

struct Reduction {
    Tensor data;
    std::int64_t chunk_size;
    std::int64_t output_count;
    Shape output_shape;
};

std::vector<bool> reduced_positions(const Tensor& t, const Dims& dims) {
    std::vector<bool> reduced(static_cast<std::size_t>(t.dim()), dims.empty());
    for (const std::int64_t dim : dims) {
        if (t.dim() == 0) {
            wrap_dim(dim, 1);
            continue;
        }
        const auto position = static_cast<std::size_t>(wrap_dim(dim, t.dim()));
        KAEL_CHECK(!reduced[position], "dimension " << dim << " appears more than once in reduction");
        reduced[position] = true;
    }
    return reduced;
}

Reduction plan_reduction(const Tensor& t, const Dims& dims, bool keepdim) {
    const std::vector<bool> reduced = reduced_positions(t, dims);

    std::vector<std::int64_t> kept_first;
    std::vector<std::int64_t> reduced_last;
    Shape kept_sizes;
    Shape reduced_sizes;
    Shape output_shape;
    for (std::size_t i = 0; i < reduced.size(); ++i) {
        const auto dim = static_cast<std::int64_t>(i);
        if (reduced[i]) {
            reduced_last.push_back(dim);
            reduced_sizes.push_back(t.shape()[i]);
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            kept_first.push_back(dim);
            kept_sizes.push_back(t.shape()[i]);
            output_shape.push_back(t.shape()[i]);
        }
    }
    kept_first.insert(kept_first.end(), reduced_last.begin(), reduced_last.end());

    return {t.permute(kept_first).contiguous(), numel(reduced_sizes), numel(kept_sizes),
            std::move(output_shape)};
}

template <typename Out, typename In, typename F>
Tensor reduce_chunks(const Reduction& reduction, DType dtype, F f) {
    Tensor out = empty(reduction.output_shape, dtype);
    const In* in = reduction.data.data_ptr<In>();
    Out* results = out.data_ptr<Out>();
    for (std::int64_t i = 0; i < reduction.output_count; ++i) {
        results[i] = f(in + i * reduction.chunk_size, reduction.chunk_size);
    }
    return out;
}

template <typename In>
using Accumulator = std::conditional_t<std::is_floating_point_v<In>, double, std::int64_t>;

template <typename In>
using AccumulationResult = std::conditional_t<std::is_floating_point_v<In>, In, std::int64_t>;

template <typename Combine>
Tensor accumulate(const Tensor& t, const Dims& dims, bool keepdim, int initial, Combine combine) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    const DType dtype = is_floating_point(t.dtype()) ? t.dtype() : DType::I64;
    return dispatch(t.dtype(), [&]<typename In>() {
        using Result = AccumulationResult<In>;
        return reduce_chunks<Result, In>(reduction, dtype, [&](const In* chunk, std::int64_t size) {
            auto total = static_cast<Accumulator<In>>(initial);
            for (std::int64_t i = 0; i < size; ++i) {
                total = combine(total, static_cast<Accumulator<In>>(chunk[i]));
            }
            return static_cast<Result>(total);
        });
    });
}

template <typename Pick>
Tensor extreme(const Tensor& t, const Dims& dims, bool keepdim, const char* op, Pick pick) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    KAEL_CHECK(reduction.chunk_size > 0, op << " of an empty dimension in shape "
                                            << to_string(t.shape()) << " has no result");
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [&](const T* chunk, std::int64_t size) {
            return std::accumulate(chunk + 1, chunk + size, chunk[0], pick);
        });
    });
}

template <typename T, typename Better>
std::int64_t best_position(const T* chunk, std::int64_t size, Better better) {
    std::int64_t best = 0;
    for (std::int64_t i = 1; i < size; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(chunk[best])) {
                break;
            }
            if (std::isnan(chunk[i])) {
                best = i;
                continue;
            }
        }
        if (better(chunk[i], chunk[best])) {
            best = i;
        }
    }
    return best;
}

template <template <typename> typename Better>
Tensor arg_extreme(const Tensor& t, const Dims& dims, bool keepdim, const char* op) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    KAEL_CHECK(reduction.chunk_size > 0, op << " of an empty dimension in shape "
                                            << to_string(t.shape()) << " has no result");
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<std::int64_t, T>(reduction, DType::I64,
                                              [](const T* chunk, std::int64_t size) {
                                                  return best_position(chunk, size, Better<T>{});
                                              });
    });
}

template <typename T>
double chunk_mean(const T* chunk, std::int64_t size) {
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        total += static_cast<double>(chunk[i]);
    }
    return total / static_cast<double>(size);
}

template <typename Predicate>
Tensor logical_reduction(const Tensor& t, const Dims& dims, bool keepdim, Predicate predicate) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<std::uint8_t, T>(reduction, DType::Bool,
                                              [&](const T* chunk, std::int64_t size) {
                                                  return predicate(chunk, chunk + size)
                                                             ? std::uint8_t{1}
                                                             : std::uint8_t{0};
                                              });
    });
}

constexpr auto kIsNonZero = [](auto value) { return value != decltype(value){0}; };

} // namespace

Tensor sum(const Tensor& t, const Dims& dims, bool keepdim) {
    return accumulate(t, dims, keepdim, 0, std::plus<>{});
}

Tensor prod(const Tensor& t, const Dims& dims, bool keepdim) {
    return accumulate(t, dims, keepdim, 1, std::multiplies<>{});
}

Tensor mean(const Tensor& t, const Dims& dims, bool keepdim) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch_floating(t.dtype(), "mean", [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [](const T* chunk, std::int64_t size) {
            return static_cast<T>(chunk_mean(chunk, size));
        });
    });
}

Tensor max(const Tensor& t, const Dims& dims, bool keepdim) {
    return extreme(t, dims, keepdim, "max", [](auto a, auto b) { return detail::nan_aware_max(a, b); });
}

Tensor min(const Tensor& t, const Dims& dims, bool keepdim) {
    return extreme(t, dims, keepdim, "min", [](auto a, auto b) { return detail::nan_aware_min(a, b); });
}

Tensor argmax(const Tensor& t) {
    return arg_extreme<std::greater>(t.flatten(), {0}, false, "argmax");
}

Tensor argmax(const Tensor& t, std::int64_t dim, bool keepdim) {
    return arg_extreme<std::greater>(t, {dim}, keepdim, "argmax");
}

Tensor argmin(const Tensor& t) {
    return arg_extreme<std::less>(t.flatten(), {0}, false, "argmin");
}

Tensor argmin(const Tensor& t, std::int64_t dim, bool keepdim) {
    return arg_extreme<std::less>(t, {dim}, keepdim, "argmin");
}

Tensor var(const Tensor& t, const Dims& dims, bool keepdim, std::int64_t correction) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch_floating(t.dtype(), "var", [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [&](const T* chunk, std::int64_t size) {
            const double center = chunk_mean(chunk, size);
            double squares = 0.0;
            for (std::int64_t i = 0; i < size; ++i) {
                const double deviation = static_cast<double>(chunk[i]) - center;
                squares += deviation * deviation;
            }
            const auto divisor = static_cast<double>(std::max<std::int64_t>(size - correction, 0));
            return static_cast<T>(squares / divisor);
        });
    });
}

Tensor std(const Tensor& t, const Dims& dims, bool keepdim, std::int64_t correction) {
    return sqrt(var(t, dims, keepdim, correction));
}

Tensor all(const Tensor& t, const Dims& dims, bool keepdim) {
    return logical_reduction(t, dims, keepdim, [](auto first, auto last) {
        return std::all_of(first, last, kIsNonZero);
    });
}

Tensor any(const Tensor& t, const Dims& dims, bool keepdim) {
    return logical_reduction(t, dims, keepdim, [](auto first, auto last) {
        return std::any_of(first, last, kIsNonZero);
    });
}

Tensor norm(const Tensor& t, const Dims& dims, bool keepdim) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch_floating(t.dtype(), "norm", [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [](const T* chunk, std::int64_t size) {
            double squares = 0.0;
            for (std::int64_t i = 0; i < size; ++i) {
                squares += static_cast<double>(chunk[i]) * static_cast<double>(chunk[i]);
            }
            return static_cast<T>(std::sqrt(squares));
        });
    });
}

Tensor Tensor::sum(const Dims& dims, bool keepdim) const {
    return kael::sum(*this, dims, keepdim);
}

Tensor Tensor::mean(const Dims& dims, bool keepdim) const {
    return kael::mean(*this, dims, keepdim);
}

} // namespace kael
