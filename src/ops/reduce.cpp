#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <type_traits>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/parallel.h"

#include "autograd/record.h"
#include "internal/nan_aware.h"

namespace kael {

namespace {

constexpr std::int64_t kReductionBlock = 16384;

// The blocks do not depend on the thread count, so floating-point results are reproducible.
template <typename Value, typename Block, typename Combine>
Value reduce_in_blocks(std::int64_t size, Value initial, Block block, Combine combine) {
    if (size <= kReductionBlock) {
        return combine(initial, block(0, size));
    }
    const std::int64_t block_count = (size + kReductionBlock - 1) / kReductionBlock;
    std::vector<Value> partial(static_cast<std::size_t>(block_count));
    parallel_for(0, block_count, kElementGrain / kReductionBlock,
                 [&](std::int64_t first, std::int64_t last) {
                     for (std::int64_t i = first; i < last; ++i) {
                         const std::int64_t begin = i * kReductionBlock;
                         partial[static_cast<std::size_t>(i)] =
                             block(begin, std::min(begin + kReductionBlock, size));
                     }
                 });
    for (std::size_t width = partial.size(); width > 1; width = (width + 1) / 2) {
        for (std::size_t i = 0; i < width / 2; ++i) {
            partial[i] = combine(partial[2 * i], partial[2 * i + 1]);
        }
        if (width % 2 == 1) {
            partial[width / 2] = partial[width - 1];
        }
    }
    return combine(initial, partial.front());
}

template <typename F> double sum_in_blocks(std::int64_t size, F term) {
    return reduce_in_blocks(
        size, 0.0,
        [&](std::int64_t first, std::int64_t last) {
            double total = 0.0;
            for (std::int64_t i = first; i < last; ++i) {
                total += term(i);
            }
            return total;
        },
        std::plus<>{});
}

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
        const std::size_t position = dim_position(dim, t.dim());
        KAEL_CHECK(!reduced[position],
                   "dimension " << dim << " appears more than once in reduction");
        reduced[position] = true;
    }
    return reduced;
}

Reduction plan_reduction(const Tensor& t, const Dims& dims, bool keepdim) {
    const autograd::NoGradGuard guard;
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

Reduction plan_nonempty_reduction(const Tensor& t, const Dims& dims, bool keepdim, const char* op) {
    Reduction reduction = plan_reduction(t, dims, keepdim);
    KAEL_CHECK(reduction.chunk_size > 0, op << " of an empty dimension in shape "
                                            << to_string(t.shape()) << " has no result");
    return reduction;
}

template <typename Out, typename In, typename F>
Tensor reduce_chunks(const Reduction& reduction, DType dtype, F f) {
    Tensor out = empty(reduction.output_shape, dtype);
    const In* in = reduction.data.data_ptr<In>();
    Out* results = out.data_ptr<Out>();
    const std::int64_t grain =
        std::max<std::int64_t>(1, kElementGrain / std::max<std::int64_t>(1, reduction.chunk_size));
    parallel_for(0, reduction.output_count, grain, [&](std::int64_t first, std::int64_t last) {
        for (std::int64_t i = first; i < last; ++i) {
            results[i] = f(in + i * reduction.chunk_size, reduction.chunk_size);
        }
    });
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
            const auto identity = static_cast<Accumulator<In>>(initial);
            return static_cast<Result>(reduce_in_blocks(
                size, identity,
                [&](std::int64_t first, std::int64_t last) {
                    auto total = identity;
                    for (std::int64_t i = first; i < last; ++i) {
                        total = combine(total, static_cast<Accumulator<In>>(chunk[i]));
                    }
                    return total;
                },
                combine));
        });
    });
}

template <typename Pick>
Tensor extreme(const Tensor& t, const Dims& dims, bool keepdim, const char* op, Pick pick) {
    const Reduction reduction = plan_nonempty_reduction(t, dims, keepdim, op);
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
    const Reduction reduction = plan_nonempty_reduction(t, dims, keepdim, op);
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<std::int64_t, T>(reduction, DType::I64,
                                              [](const T* chunk, std::int64_t size) {
                                                  return best_position(chunk, size, Better<T>{});
                                              });
    });
}

template <typename T> double chunk_mean(const T* chunk, std::int64_t size) {
    const double total =
        sum_in_blocks(size, [&](std::int64_t i) { return static_cast<double>(chunk[i]); });
    return total / static_cast<double>(size);
}

template <typename F>
Tensor reduce_floating(const Tensor& t, const Dims& dims, bool keepdim, const char* op,
                       F chunk_value) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch_floating(t.dtype(), op, [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [&](const T* chunk, std::int64_t size) {
            return static_cast<T>(chunk_value(chunk, size));
        });
    });
}

template <typename Predicate>
Tensor logical_reduction(const Tensor& t, const Dims& dims, bool keepdim, Predicate predicate) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<std::uint8_t, T>(
            reduction, DType::Bool, [&](const T* chunk, std::int64_t size) {
                return predicate(chunk, chunk + size) ? std::uint8_t{1} : std::uint8_t{0};
            });
    });
}

constexpr auto kIsNonZero = [](auto value) { return value != decltype(value){0}; };

} // namespace

Tensor sum(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = accumulate(t, dims, keepdim, 0, std::plus<>{});
    autograd::record_sum(out, t, dims, keepdim);
    return out;
}

Tensor prod(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = accumulate(t, dims, keepdim, 1, std::multiplies<>{});
    autograd::record_prod(out, t, dims, keepdim);
    return out;
}

Tensor mean(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out =
        reduce_floating(t, dims, keepdim, "mean", [](const auto* chunk, std::int64_t size) {
            return chunk_mean(chunk, size);
        });
    autograd::record_mean(out, t, dims, keepdim);
    return out;
}

Tensor max(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = extreme(t, dims, keepdim, "max",
                         [](auto a, auto b) { return detail::nan_aware_max(a, b); });
    autograd::record_extreme(out, t, dims, keepdim, "MaxBackward");
    return out;
}

Tensor min(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = extreme(t, dims, keepdim, "min",
                         [](auto a, auto b) { return detail::nan_aware_min(a, b); });
    autograd::record_extreme(out, t, dims, keepdim, "MinBackward");
    return out;
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
    Tensor out =
        reduce_floating(t, dims, keepdim, "var", [&](const auto* chunk, std::int64_t size) {
            const double center = chunk_mean(chunk, size);
            const double squares = sum_in_blocks(size, [&](std::int64_t i) {
                const double deviation = static_cast<double>(chunk[i]) - center;
                return deviation * deviation;
            });
            return squares / static_cast<double>(std::max<std::int64_t>(size - correction, 0));
        });
    autograd::record_var(out, t, dims, keepdim, correction);
    return out;
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
    Tensor out =
        reduce_floating(t, dims, keepdim, "norm", [](const auto* chunk, std::int64_t size) {
            return std::sqrt(sum_in_blocks(size, [&](std::int64_t i) {
                return static_cast<double>(chunk[i]) * static_cast<double>(chunk[i]);
            }));
        });
    autograd::record_norm(out, t, dims, keepdim);
    return out;
}

Tensor Tensor::sum(const Dims& dims, bool keepdim) const {
    return kael::sum(*this, dims, keepdim);
}

Tensor Tensor::mean(const Dims& dims, bool keepdim) const {
    return kael::mean(*this, dims, keepdim);
}

} // namespace kael
