// Included by the CUDA backend and by the host emulation, which provide for_each and atomic_add.

namespace kael::device::tasks {

template <typename To, typename From> KAEL_HD To cast_value(From value) {
    if constexpr (std::is_same_v<To, std::uint8_t>) {
        return value != From{0} ? std::uint8_t{1} : std::uint8_t{0};
    } else {
        return static_cast<To>(value);
    }
}

KAEL_HD inline void locate(const ElementLayout& layout, std::int64_t element, std::byte** pointers) {
    std::int64_t offsets[kMaxOperands] = {};
    std::int64_t remaining = element;
    for (int dim = layout.dims - 1; dim >= 0; --dim) {
        const std::int64_t position = remaining % layout.shape[dim];
        remaining /= layout.shape[dim];
        for (int op = 0; op < layout.operands; ++op) {
            offsets[op] += position * layout.strides[op][dim];
        }
    }
    for (int op = 0; op < layout.operands; ++op) {
        pointers[op] = layout.base[op] + offsets[op];
    }
}

template <typename Out, typename In, int Inputs, typename F> struct Element {
    ElementLayout layout;
    F function;

    KAEL_HD void operator()(std::int64_t element) const {
        std::byte* pointers[kMaxOperands];
        locate(layout, element, pointers);
        const auto read = [&](int op) { return *reinterpret_cast<const In*>(pointers[op]); };
        Out& out = *reinterpret_cast<Out*>(pointers[0]);
        if constexpr (Inputs == 1) {
            out = cast_value<Out>(function(read(1)));
        } else if constexpr (Inputs == 2) {
            out = cast_value<Out>(function(read(1), read(2)));
        } else {
            out = cast_value<Out>(function(read(1), read(2), read(3)));
        }
    }
};

struct Identity {
    template <typename T> KAEL_HD T operator()(T x) const { return x; }
};

template <typename T> struct Fill {
    T* target;
    T value;
    KAEL_HD void operator()(std::int64_t element) const { target[element] = value; }
};

template <typename In, typename Accumulator> KAEL_HD Accumulator sum_range(const In* data, std::int64_t begin, std::int64_t end) {
    Accumulator total{0};
    for (std::int64_t i = begin; i < end; ++i) {
        total += static_cast<Accumulator>(data[i]);
    }
    return total;
}

template <ReduceOp Op> struct Combine;

template <> struct Combine<ReduceOp::Sum> {
    template <typename T> KAEL_HD T operator()(T a, T b) const { return a + b; }
};
template <> struct Combine<ReduceOp::Max> {
    template <typename T> KAEL_HD T operator()(T a, T b) const {
        return functions::nan_aware_max(a, b);
    }
};
template <> struct Combine<ReduceOp::Min> {
    template <typename T> KAEL_HD T operator()(T a, T b) const {
        return functions::nan_aware_min(a, b);
    }
};

template <ReduceOp Op, typename In, typename Accumulator>
KAEL_HD Accumulator reduce_range(const In* data, std::int64_t begin, std::int64_t end) {
    if constexpr (Op == ReduceOp::Sum) {
        return sum_range<In, Accumulator>(data, begin, end);
    } else {
        Accumulator best = static_cast<Accumulator>(data[begin]);
        for (std::int64_t i = begin + 1; i < end; ++i) {
            best = Combine<Op>{}(best, static_cast<Accumulator>(data[i]));
        }
        return best;
    }
}

template <ReduceOp Op, typename Out, typename In, typename Accumulator> struct ReduceRows {
    Out* results;
    const In* data;
    std::int64_t chunk_size;
    KAEL_HD void operator()(std::int64_t row) const {
        results[row] = static_cast<Out>(
            reduce_range<Op, In, Accumulator>(data + row * chunk_size, 0, chunk_size));
    }
};

template <ReduceOp Op, typename In, typename Accumulator> struct ReduceBlocks {
    Accumulator* partial;
    const In* data;
    std::int64_t block_size;
    std::int64_t total;
    KAEL_HD void operator()(std::int64_t block) const {
        const std::int64_t begin = block * block_size;
        const std::int64_t end = begin + block_size < total ? begin + block_size : total;
        partial[block] = reduce_range<Op, In, Accumulator>(data, begin, end);
    }
};

template <ReduceOp Op, typename Accumulator> struct MergePairs {
    Accumulator* target;
    const Accumulator* source;
    std::int64_t width;
    KAEL_HD void operator()(std::int64_t i) const {
        target[i] = 2 * i + 1 < width ? Combine<Op>{}(source[2 * i], source[2 * i + 1])
                                      : source[2 * i];
    }
};

template <typename Out, typename Accumulator> struct StoreResult {
    Out* result;
    const Accumulator* source;
    KAEL_HD void operator()(std::int64_t) const { *result = static_cast<Out>(*source); }
};

inline constexpr std::int64_t kMatmulTile = 4;

template <typename T> struct MatmulTiles {
    const T* a;
    const T* b;
    T* out;
    MatmulSizes sizes;
    std::int64_t tile_rows;
    std::int64_t tile_columns;

    KAEL_HD void operator()(std::int64_t tile) const {
        const std::int64_t per_batch = tile_rows * tile_columns;
        const std::int64_t batch = tile / per_batch;
        const std::int64_t in_batch = tile % per_batch;
        const std::int64_t first_row = (in_batch / tile_columns) * kMatmulTile;
        const std::int64_t first_column = (in_batch % tile_columns) * kMatmulTile;
        const std::int64_t row_count =
            sizes.rows - first_row < kMatmulTile ? sizes.rows - first_row : kMatmulTile;
        const std::int64_t column_count =
            sizes.columns - first_column < kMatmulTile ? sizes.columns - first_column : kMatmulTile;

        const T* batch_a = a + batch * sizes.rows * sizes.inner;
        const T* batch_b = b + batch * sizes.inner * sizes.columns;
        T accumulators[kMatmulTile][kMatmulTile] = {};
        for (std::int64_t k = 0; k < sizes.inner; ++k) {
            const T* b_row = batch_b + k * sizes.columns + first_column;
            for (std::int64_t r = 0; r < row_count; ++r) {
                const T a_value = batch_a[(first_row + r) * sizes.inner + k];
                for (std::int64_t c = 0; c < column_count; ++c) {
                    accumulators[r][c] = static_cast<T>(accumulators[r][c] + a_value * b_row[c]);
                }
            }
        }
        T* batch_out = out + batch * sizes.rows * sizes.columns;
        for (std::int64_t r = 0; r < row_count; ++r) {
            for (std::int64_t c = 0; c < column_count; ++c) {
                batch_out[(first_row + r) * sizes.columns + first_column + c] = accumulators[r][c];
            }
        }
    }
};

struct IndexPosition {
    std::int64_t data = 0;
    std::int64_t index = 0;
    std::int64_t source = 0;
};

KAEL_HD inline IndexPosition locate_index(const IndexLayout& layout, const std::int64_t* indices,
                                          std::int64_t element) {
    IndexPosition position;
    std::int64_t remaining = element;
    for (int dim = layout.dims - 1; dim >= 0; --dim) {
        const std::int64_t coordinate = remaining % layout.shape[dim];
        remaining /= layout.shape[dim];
        position.index += coordinate * layout.index_strides[dim];
        position.source += coordinate * layout.source_strides[dim];
        if (dim != layout.axis) {
            position.data += coordinate * layout.data_strides[dim];
        }
    }
    position.data += indices[position.index] * layout.data_strides[layout.axis];
    return position;
}

template <typename T> struct Gather {
    T* out;
    const T* data;
    const std::int64_t* indices;
    IndexLayout layout;
    KAEL_HD void operator()(std::int64_t element) const {
        out[element] = data[locate_index(layout, indices, element).data];
    }
};

template <typename T> struct ScatterAdd {
    T* data;
    const std::int64_t* indices;
    const T* source;
    IndexLayout layout;
    KAEL_HD void operator()(std::int64_t element) const {
        const IndexPosition position = locate_index(layout, indices, element);
        atomic_add(data + position.data, source[position.source]);
    }
};

} // namespace kael::device::tasks

namespace kael::device {

namespace {

template <DType D> struct Tag {
    using Type = cpp_type_t<D>;
};

template <typename F> void with_type(DType dtype, F f) {
    switch (dtype) {
    case DType::F32:
        f(float{});
        return;
    case DType::F64:
        f(double{});
        return;
    case DType::I64:
        f(std::int64_t{});
        return;
    case DType::Bool:
        f(std::uint8_t{});
        return;
    }
}

template <typename F> void with_floating_type(DType dtype, F f) {
    if (dtype == DType::F32) {
        f(float{});
    } else {
        f(double{});
    }
}

template <int Inputs, typename Function>
void run_same(DType in, const ElementLayout& layout, Function function) {
    with_type(in, [&](auto sample) {
        using T = decltype(sample);
        for_each(layout.count, tasks::Element<T, T, Inputs, Function>{layout, function});
    });
}

template <typename Function>
void run_floating(DType in, const ElementLayout& layout, Function function) {
    with_floating_type(in, [&](auto sample) {
        using T = decltype(sample);
        for_each(layout.count, tasks::Element<T, T, 1, Function>{layout, function});
    });
}

template <typename Function>
void run_compare(DType in, const ElementLayout& layout, Function function) {
    with_type(in, [&](auto sample) {
        using T = decltype(sample);
        for_each(layout.count, tasks::Element<std::uint8_t, T, 2, Function>{layout, function});
    });
}

template <ReduceOp Op, typename Out, typename In, typename Accumulator>
void reduce_typed(Out* results, const In* data, std::int64_t chunk_size,
                  std::int64_t output_count) {
    constexpr std::int64_t kBlock = 4096;
    if (chunk_size <= kBlock) {
        for_each(output_count, tasks::ReduceRows<Op, Out, In, Accumulator>{results, data, chunk_size});
        return;
    }
    const std::int64_t block_count = (chunk_size + kBlock - 1) / kBlock;
    const auto bytes = block_count * static_cast<std::int64_t>(sizeof(Accumulator));
    std::byte* first = allocate(bytes);
    std::byte* second = allocate(bytes);
    auto* partial = reinterpret_cast<Accumulator*>(first);
    auto* merged = reinterpret_cast<Accumulator*>(second);
    for (std::int64_t row = 0; row < output_count; ++row) {
        for_each(block_count, tasks::ReduceBlocks<Op, In, Accumulator>{
                                  partial, data + row * chunk_size, kBlock, chunk_size});
        Accumulator* source = partial;
        Accumulator* target = merged;
        for (std::int64_t width = block_count; width > 1; width = (width + 1) / 2) {
            for_each((width + 1) / 2, tasks::MergePairs<Op, Accumulator>{target, source, width});
            Accumulator* const previous = source;
            source = target;
            target = previous;
        }
        for_each(1, tasks::StoreResult<Out, Accumulator>{results + row, source});
    }
    release(second);
    release(first);
}

template <typename T> void scatter_typed(std::byte* data, const std::int64_t* indices, const std::byte* source, const IndexLayout& layout) {
    for_each(layout.count, tasks::ScatterAdd<T>{reinterpret_cast<T*>(data), indices,
                                                reinterpret_cast<const T*>(source), layout});
}

} // namespace

void fill(DType dtype, std::byte* target, std::int64_t count, const void* value) {
    with_type(dtype, [&](auto sample) {
        using T = decltype(sample);
        for_each(count, tasks::Fill<T>{reinterpret_cast<T*>(target), *static_cast<const T*>(value)});
    });
}

void convert(DType out, DType in, const ElementLayout& layout) {
    with_type(out, [&](auto out_sample) {
        with_type(in, [&](auto in_sample) {
            using Out = decltype(out_sample);
            using In = decltype(in_sample);
            for_each(layout.count, tasks::Element<Out, In, 1, tasks::Identity>{layout, {}});
        });
    });
}

void elementwise(ElementOp op, DType out, DType in, const ElementLayout& layout,
                 ElementParameters parameters) {
    (void)out;
    switch (op) {
    case ElementOp::Plus:
        return run_same<2>(in, layout, functions::Plus{});
    case ElementOp::Minus:
        return run_same<2>(in, layout, functions::Minus{});
    case ElementOp::Times:
        return run_same<2>(in, layout, functions::Times{});
    case ElementOp::Divides:
        return run_same<2>(in, layout, functions::Divides{});
    case ElementOp::Power:
        return with_type(in, [&](auto sample) {
            using T = decltype(sample);
            if constexpr (!std::is_same_v<T, std::uint8_t>) {
                for_each(layout.count, tasks::Element<T, T, 2, functions::Power>{layout, {}});
            }
        });
    case ElementOp::Maximum:
        return run_same<2>(in, layout, functions::Maximum{});
    case ElementOp::Minimum:
        return run_same<2>(in, layout, functions::Minimum{});
    case ElementOp::Equal:
        return run_compare(in, layout, functions::Equal{});
    case ElementOp::NotEqual:
        return run_compare(in, layout, functions::NotEqual{});
    case ElementOp::Less:
        return run_compare(in, layout, functions::Less{});
    case ElementOp::LessEqual:
        return run_compare(in, layout, functions::LessEqual{});
    case ElementOp::Greater:
        return run_compare(in, layout, functions::Greater{});
    case ElementOp::GreaterEqual:
        return run_compare(in, layout, functions::GreaterEqual{});
    case ElementOp::LogicalAnd:
        return run_compare(in, layout, functions::LogicalAnd{});
    case ElementOp::LogicalOr:
        return run_compare(in, layout, functions::LogicalOr{});
    case ElementOp::LogicalXor:
        return run_compare(in, layout, functions::LogicalXor{});
    case ElementOp::IsZero:
        return run_same<1>(in, layout, functions::IsZero{});
    case ElementOp::Negate:
        return run_same<1>(in, layout, functions::Negate{});
    case ElementOp::Absolute:
        return run_same<1>(in, layout, functions::Absolute{});
    case ElementOp::Sign:
        return run_same<1>(in, layout, functions::Sign{});
    case ElementOp::Exp:
        return run_floating(in, layout, functions::Exp{});
    case ElementOp::Log:
        return run_floating(in, layout, functions::Log{});
    case ElementOp::Log1p:
        return run_floating(in, layout, functions::Log1p{});
    case ElementOp::Expm1:
        return run_floating(in, layout, functions::Expm1{});
    case ElementOp::Sqrt:
        return run_floating(in, layout, functions::Sqrt{});
    case ElementOp::Rsqrt:
        return run_floating(in, layout, functions::Rsqrt{});
    case ElementOp::Sin:
        return run_floating(in, layout, functions::Sin{});
    case ElementOp::Cos:
        return run_floating(in, layout, functions::Cos{});
    case ElementOp::Tanh:
        return run_floating(in, layout, functions::Tanh{});
    case ElementOp::Sigmoid:
        return run_floating(in, layout, functions::Sigmoid{});
    case ElementOp::Relu:
        return run_same<1>(in, layout, functions::Relu{});
    case ElementOp::GeluExact:
        return run_floating(in, layout, functions::GeluExact{});
    case ElementOp::GeluTanh:
        return run_floating(in, layout, functions::GeluTanh{});
    case ElementOp::Clamp:
        return with_type(in, [&](auto sample) {
            using T = decltype(sample);
            const functions::Clamp<T> clamp{static_cast<T>(parameters.first),
                                            static_cast<T>(parameters.second)};
            for_each(layout.count, tasks::Element<T, T, 1, functions::Clamp<T>>{layout, clamp});
        });
    case ElementOp::Where:
        return run_same<3>(in, layout, functions::Where{});
    case ElementOp::MaskedFill:
        return with_type(in, [&](auto sample) {
            using T = decltype(sample);
            const functions::MaskedFill<T> fill_mask{static_cast<T>(parameters.first)};
            for_each(layout.count,
                     tasks::Element<T, T, 2, functions::MaskedFill<T>>{layout, fill_mask});
        });
    }
}

void reduce(ReduceOp op, DType out, DType in, std::byte* results, const std::byte* data,
            std::int64_t chunk_size, std::int64_t output_count) {
    with_type(in, [&](auto sample) {
        using In = decltype(sample);
        const auto* typed = reinterpret_cast<const In*>(data);
        if (op == ReduceOp::Sum) {
            using Accumulator = std::conditional_t<std::is_floating_point_v<In>, double, std::int64_t>;
            using Out = std::conditional_t<std::is_floating_point_v<In>, In, std::int64_t>;
            reduce_typed<ReduceOp::Sum, Out, In, Accumulator>(reinterpret_cast<Out*>(results), typed,
                                                              chunk_size, output_count);
        } else if (op == ReduceOp::Max) {
            reduce_typed<ReduceOp::Max, In, In, In>(reinterpret_cast<In*>(results), typed, chunk_size,
                                                    output_count);
        } else {
            reduce_typed<ReduceOp::Min, In, In, In>(reinterpret_cast<In*>(results), typed, chunk_size,
                                                    output_count);
        }
    });
    (void)out;
}

void gather(DType dtype, std::byte* out, const std::byte* data, const std::int64_t* indices,
            const IndexLayout& layout) {
    with_type(dtype, [&](auto sample) {
        using T = decltype(sample);
        for_each(layout.count, tasks::Gather<T>{reinterpret_cast<T*>(out),
                                                reinterpret_cast<const T*>(data), indices, layout});
    });
}

void scatter_add(DType dtype, std::byte* data, const std::int64_t* indices, const std::byte* source,
                 const IndexLayout& layout) {
    if (dtype == DType::F32) {
        scatter_typed<float>(data, indices, source, layout);
    } else if (dtype == DType::F64) {
        scatter_typed<double>(data, indices, source, layout);
    } else {
        scatter_typed<std::int64_t>(data, indices, source, layout);
    }
}

} // namespace kael::device
