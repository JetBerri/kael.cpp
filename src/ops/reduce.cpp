#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <type_traits>

#include "kael/autograd/grad_mode.h"
#include "kael/device_kernels.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/parallel.h"

#include "autograd/record.h"
#include "internal/element_functions.h"
#include "internal/nan_aware.h"

namespace kael {

namespace {

constexpr std::int64_t kReductionBlock = 16384;
constexpr std::int64_t kLanes = 8;

template <typename T> struct StridedChunk {
    const T* base;
    std::int64_t stride;

    const T& operator[](std::int64_t i) const { return base[i * stride]; }
};

// Independent accumulators let the compiler vectorize without reordering a single accumulator.
template <typename Value, typename Term, typename Combine>
Value fold_lanes(std::int64_t first, std::int64_t last, Value identity, Term term,
                 Combine combine) {
    std::array<Value, kLanes> lanes;
    lanes.fill(identity);
    std::int64_t i = first;
    for (; i + kLanes <= last; i += kLanes) {
        for (std::int64_t lane = 0; lane < kLanes; ++lane) {
            auto& accumulator = lanes[static_cast<std::size_t>(lane)];
            accumulator = combine(accumulator, term(i + lane));
        }
    }
    Value total = identity;
    for (const Value lane : lanes) {
        total = combine(total, lane);
    }
    for (; i < last; ++i) {
        total = combine(total, term(i));
    }
    return total;
}

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
            return fold_lanes(first, last, 0.0, term, std::plus<>{});
        },
        std::plus<>{});
}

struct Reduction {
    Tensor data;
    std::int64_t chunk_size;
    std::int64_t output_count;
    Shape output_shape;
    std::int64_t chunk_stride = 1;
    Shape kept_sizes = {};
    Strides kept_strides = {};
};

std::vector<bool> reduced_positions(const Tensor& t, const Dims& dims) {
    std::vector<bool> reduced(static_cast<std::size_t>(t.dim()), dims.empty());
    for (const std::int64_t dim : dims) {
        if (t.dim() == 0) {
            wrap_dim(dim, 1);
            continue;
        }
        const std::size_t position = dim_position(dim, t.dim());
        KAEL_CHECK_VALUE(!reduced[position],
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

    Tensor permuted = t.permute(kept_first);
    if (!permuted.is_contiguous() && reduced_sizes.size() == 1 && t.device() == Device::CPU) {
        const std::int64_t chunk_stride = permuted.strides().back();
        Strides kept_strides(permuted.strides().begin(), permuted.strides().end() - 1);
        return {std::move(permuted),     reduced_sizes.front(), numel(kept_sizes),
                std::move(output_shape), chunk_stride,          std::move(kept_sizes),
                std::move(kept_strides)};
    }
    return {permuted.contiguous(), numel(reduced_sizes), numel(kept_sizes),
            std::move(output_shape)};
}

Reduction plan_nonempty_reduction(const Tensor& t, const Dims& dims, bool keepdim, const char* op) {
    Reduction reduction = plan_reduction(t, dims, keepdim);
    KAEL_CHECK_VALUE(reduction.chunk_size > 0, op << " of an empty dimension in shape "
                                                  << to_string(t.shape()) << " has no result");
    return reduction;
}

std::int64_t kept_offset(const Reduction& reduction, std::int64_t output) {
    std::int64_t offset = 0;
    for (std::size_t dim = reduction.kept_sizes.size(); dim-- > 0;) {
        offset += output % reduction.kept_sizes[dim] * reduction.kept_strides[dim];
        output /= reduction.kept_sizes[dim];
    }
    return offset;
}

template <typename Out, typename In, typename F>
Tensor reduce_chunks(const Reduction& reduction, DType dtype, F f) {
    Tensor out = empty(reduction.output_shape, dtype);
    const In* in = reduction.data.data_ptr<In>();
    Out* results = out.data_ptr<Out>();
    const std::int64_t size = reduction.chunk_size;
    const std::int64_t grain =
        std::max<std::int64_t>(1, kElementGrain / std::max<std::int64_t>(1, size));
    const bool strided = !reduction.kept_sizes.empty() || reduction.chunk_stride != 1;
    parallel_for(0, reduction.output_count, grain, [&](std::int64_t first, std::int64_t last) {
        for (std::int64_t i = first; i < last; ++i) {
            if (!strided) {
                results[i] = f(in + i * size, size);
                continue;
            }
            const In* base = in + kept_offset(reduction, i);
            results[i] = reduction.chunk_stride == 1
                             ? f(base, size)
                             : f(StridedChunk<In>{base, reduction.chunk_stride}, size);
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
        return reduce_chunks<Result, In>(reduction, dtype, [&](auto chunk, std::int64_t size) {
            const auto identity = static_cast<Accumulator<In>>(initial);
            return static_cast<Result>(reduce_in_blocks(
                size, identity,
                [&](std::int64_t first, std::int64_t last) {
                    return fold_lanes(
                        first, last, identity,
                        [&](std::int64_t i) { return static_cast<Accumulator<In>>(chunk[i]); },
                        combine);
                },
                combine));
        });
    });
}

template <typename Pick>
Tensor extreme(const Tensor& t, const Dims& dims, bool keepdim, const char* op, Pick pick) {
    const Reduction reduction = plan_nonempty_reduction(t, dims, keepdim, op);
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [&](auto chunk, std::int64_t size) {
            T best = chunk[0];
            for (std::int64_t i = 1; i < size; ++i) {
                best = pick(best, chunk[i]);
            }
            return best;
        });
    });
}

template <typename T, typename Chunk, typename Better>
std::int64_t best_position(Chunk chunk, std::int64_t size, Better better) {
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
                                              [](auto chunk, std::int64_t size) {
                                                  return best_position<T>(chunk, size, Better<T>{});
                                              });
    });
}

template <typename Chunk> double chunk_mean(Chunk chunk, std::int64_t size) {
    const double total =
        sum_in_blocks(size, [&](std::int64_t i) { return static_cast<double>(chunk[i]); });
    return total / static_cast<double>(size);
}

template <typename F>
Tensor reduce_floating(const Tensor& t, const Dims& dims, bool keepdim, const char* op,
                       F chunk_value) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch_floating(t.dtype(), op, [&]<typename T>() {
        return reduce_chunks<T, T>(reduction, t.dtype(), [&](auto chunk, std::int64_t size) {
            return static_cast<T>(chunk_value(chunk, size));
        });
    });
}

template <typename Predicate>
Tensor logical_reduction(const Tensor& t, const Dims& dims, bool keepdim, Predicate predicate) {
    const Reduction reduction = plan_reduction(t, dims, keepdim);
    return dispatch(t.dtype(), [&]<typename T>() {
        return reduce_chunks<std::uint8_t, T>(
            reduction, DType::Bool, [&](auto chunk, std::int64_t size) {
                return predicate(chunk, size) ? std::uint8_t{1} : std::uint8_t{0};
            });
    });
}

constexpr auto kIsNonZero = [](auto value) { return value != decltype(value){0}; };

bool on_device(const Tensor& t) {
    return t.device() != Device::CPU;
}

template <typename Compute> Tensor on_host(const Tensor& t, Compute compute) {
    if (!on_device(t)) {
        return compute(t);
    }
    const autograd::NoGradGuard guard;
    return compute(t.cpu()).to(t.device());
}

Tensor device_reduce(const Reduction& reduction, device::ReduceOp op, DType dtype) {
    Tensor out = empty(reduction.output_shape, dtype, reduction.data.device());
    device::reduce(op, dtype, reduction.data.dtype(), out.device_data(),
                   reduction.data.device_data(), reduction.chunk_size, reduction.output_count);
    return out;
}

Tensor device_sum(const Tensor& t, const Dims& dims, bool keepdim) {
    return device_reduce(plan_reduction(t, dims, keepdim), device::ReduceOp::Sum,
                         is_floating_point(t.dtype()) ? t.dtype() : DType::I64);
}

std::int64_t reduced_count(const Tensor& t, const Tensor& reduced) {
    return t.numel() / std::max<std::int64_t>(1, reduced.numel());
}

void check_floating(const Tensor& t, const char* op) {
    KAEL_CHECK_TYPE(is_floating_point(t.dtype()),
                    op << " only supports floating point dtypes, got " << t.dtype());
}

Tensor device_mean(const Tensor& t, const Dims& dims, bool keepdim) {
    check_floating(t, "mean");
    const autograd::NoGradGuard guard;
    const Tensor total = device_sum(t, dims, keepdim);
    return total / reduced_count(t, total);
}

Tensor device_var(const Tensor& t, const Dims& dims, bool keepdim, std::int64_t correction) {
    check_floating(t, "var");
    const autograd::NoGradGuard guard;
    const Tensor deviation = t - device_mean(t, dims, true);
    const Tensor squares = device_sum(deviation * deviation, dims, keepdim);
    return squares / std::max<std::int64_t>(reduced_count(t, squares) - correction, 0);
}

Tensor device_norm(const Tensor& t, const Dims& dims, bool keepdim) {
    check_floating(t, "norm");
    const autograd::NoGradGuard guard;
    return sqrt(device_sum(t * t, dims, keepdim));
}

} // namespace

Tensor sum(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = on_device(t) ? device_sum(t, dims, keepdim)
                              : accumulate(t, dims, keepdim, 0, [](auto a, auto b) {
                                    return functions::wrapping_add(a, b);
                                });
    autograd::record_sum(out, t, dims, keepdim);
    return out;
}

Tensor prod(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = on_host(t, [&](const Tensor& host) {
        return accumulate(host, dims, keepdim, 1,
                          [](auto a, auto b) { return functions::wrapping_multiply(a, b); });
    });
    autograd::record_prod(out, t, dims, keepdim);
    return out;
}

Tensor mean(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = on_device(t)
                     ? device_mean(t, dims, keepdim)
                     : reduce_floating(t, dims, keepdim, "mean", [](auto chunk, std::int64_t size) {
                           return chunk_mean(chunk, size);
                       });
    autograd::record_mean(out, t, dims, keepdim);
    return out;
}

Tensor max(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = on_device(t) ? device_reduce(plan_nonempty_reduction(t, dims, keepdim, "max"),
                                              device::ReduceOp::Max, t.dtype())
                              : extreme(t, dims, keepdim, "max",
                                        [](auto a, auto b) { return detail::nan_aware_max(a, b); });
    autograd::record_extreme(out, t, dims, keepdim, "MaxBackward");
    return out;
}

Tensor min(const Tensor& t, const Dims& dims, bool keepdim) {
    Tensor out = on_device(t) ? device_reduce(plan_nonempty_reduction(t, dims, keepdim, "min"),
                                              device::ReduceOp::Min, t.dtype())
                              : extreme(t, dims, keepdim, "min",
                                        [](auto a, auto b) { return detail::nan_aware_min(a, b); });
    autograd::record_extreme(out, t, dims, keepdim, "MinBackward");
    return out;
}

Tensor argmax(const Tensor& t) {
    return on_host(t, [](const Tensor& host) {
        return arg_extreme<std::greater>(host.flatten(), {0}, false, "argmax");
    });
}

Tensor argmax(const Tensor& t, std::int64_t dim, bool keepdim) {
    return on_host(t, [&](const Tensor& host) {
        return arg_extreme<std::greater>(host, {dim}, keepdim, "argmax");
    });
}

Tensor argmin(const Tensor& t) {
    return on_host(t, [](const Tensor& host) {
        return arg_extreme<std::less>(host.flatten(), {0}, false, "argmin");
    });
}

Tensor argmin(const Tensor& t, std::int64_t dim, bool keepdim) {
    return on_host(t, [&](const Tensor& host) {
        return arg_extreme<std::less>(host, {dim}, keepdim, "argmin");
    });
}

Tensor var(const Tensor& t, const Dims& dims, bool keepdim, std::int64_t correction) {
    if (on_device(t)) {
        Tensor out = device_var(t, dims, keepdim, correction);
        autograd::record_var(out, t, dims, keepdim, correction);
        return out;
    }
    Tensor out = reduce_floating(t, dims, keepdim, "var", [&](auto chunk, std::int64_t size) {
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
    return on_host(t, [&](const Tensor& host) {
        return logical_reduction(host, dims, keepdim, [](auto chunk, std::int64_t size) {
            for (std::int64_t i = 0; i < size; ++i) {
                if (!kIsNonZero(chunk[i])) {
                    return false;
                }
            }
            return true;
        });
    });
}

Tensor any(const Tensor& t, const Dims& dims, bool keepdim) {
    return on_host(t, [&](const Tensor& host) {
        return logical_reduction(host, dims, keepdim, [](auto chunk, std::int64_t size) {
            for (std::int64_t i = 0; i < size; ++i) {
                if (kIsNonZero(chunk[i])) {
                    return true;
                }
            }
            return false;
        });
    });
}

Tensor norm(const Tensor& t, const Dims& dims, bool keepdim) {
    if (on_device(t)) {
        Tensor out = device_norm(t, dims, keepdim);
        autograd::record_norm(out, t, dims, keepdim);
        return out;
    }
    Tensor out = reduce_floating(t, dims, keepdim, "norm", [](auto chunk, std::int64_t size) {
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
