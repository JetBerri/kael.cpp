#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "autograd/record.h"

namespace kael {

namespace {

template <typename T> bool ordered_before(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return !std::isnan(a) && std::isnan(b);
        }
    }
    return a < b;
}

template <typename T> auto comparator(const T* row, bool descending) {
    return [row, descending](std::int64_t a, std::int64_t b) {
        return descending ? ordered_before(row[b], row[a]) : ordered_before(row[a], row[b]);
    };
}

Tensor moved_to_last_on_host(const Tensor& t, std::int64_t dim) {
    return t.detach().cpu().transpose(dim, -1).contiguous();
}

Tensor moved_back(const Tensor& t, std::int64_t dim, Device device) {
    return t.transpose(dim, -1).contiguous().to(device);
}

Tensor sorted_positions(const Tensor& rows, std::int64_t kept, bool descending) {
    const std::int64_t size = rows.size(-1);
    const std::int64_t row_count = size > 0 ? rows.numel() / size : 0;
    Shape shape = rows.shape();
    shape.back() = kept;
    Tensor out = empty(shape, DType::I64);
    std::int64_t* results = out.data_ptr<std::int64_t>();
    std::vector<std::int64_t> order(static_cast<std::size_t>(size));
    dispatch(rows.dtype(), [&]<typename T>() {
        const T* data = rows.data_ptr<T>();
        for (std::int64_t row = 0; row < row_count; ++row) {
            std::iota(order.begin(), order.end(), std::int64_t{0});
            const auto compare = comparator(data + row * size, descending);
            if (kept == size) {
                std::stable_sort(order.begin(), order.end(), compare);
            } else {
                std::partial_sort(order.begin(), order.begin() + kept, order.end(), compare);
            }
            std::copy_n(order.begin(), kept, results + row * kept);
        }
    });
    return out;
}

SortResult select_sorted(const Tensor& t, std::int64_t kept, std::int64_t dim, bool descending) {
    const autograd::NoGradGuard outer_guard;
    const Tensor rows = moved_to_last_on_host(t, dim);
    return {t, moved_back(sorted_positions(rows, kept, descending), dim, t.device())};
}

SortResult with_values(const Tensor& t, std::int64_t dim, SortResult result) {
    return {gather(t, dim, result.indices), std::move(result.indices)};
}

Tensor triangular_mask(const Tensor& t, std::int64_t diagonal, bool lower) {
    KAEL_CHECK_VALUE(t.dim() >= 2, "tril and triu need at least 2 dimensions, got shape "
                                       << to_string(t.shape()));
    const std::int64_t rows = t.size(-2);
    const std::int64_t columns = t.size(-1);
    Tensor mask = empty({rows, columns}, DType::Bool);
    auto* flags = mask.data_ptr<std::uint8_t>();
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < columns; ++j) {
            const bool kept = lower ? j - i <= diagonal : j - i >= diagonal;
            flags[i * columns + j] = kept ? 1 : 0;
        }
    }
    return mask.to(t.device());
}

} // namespace

Tensor cumsum(const Tensor& t, std::int64_t dim) {
    if (t.dim() > 0) {
        (void)dim_position(dim, t.dim());
    }
    const DType dtype = is_floating_point(t.dtype()) ? t.dtype() : DType::I64;
    Tensor out = autograd::without_grad([&] {
        const Tensor rows = moved_to_last_on_host(t, dim).to(dtype);
        const std::int64_t size = rows.size(-1);
        Tensor totals = empty(rows.shape(), dtype);
        dispatch(dtype, [&]<typename T>() {
            const T* data = rows.data_ptr<T>();
            T* results = totals.data_ptr<T>();
            for (std::int64_t i = 0; i < rows.numel(); ++i) {
                const bool starts_row = i % size == 0;
                results[i] = starts_row ? data[i] : static_cast<T>(results[i - 1] + data[i]);
            }
        });
        return moved_back(totals, dim, t.device());
    });
    autograd::record_cumsum(out, t, dim);
    return out;
}

SortResult sort(const Tensor& t, std::int64_t dim, bool descending) {
    return with_values(t, dim, select_sorted(t, t.size(dim), dim, descending));
}

SortResult topk(const Tensor& t, std::int64_t k, std::int64_t dim, bool largest) {
    KAEL_CHECK_VALUE(k >= 0 && k <= t.size(dim),
                     "topk needs k in [0, " << t.size(dim) << "], got " << k);
    return with_values(t, dim, select_sorted(t, k, dim, largest));
}

Tensor tril(const Tensor& t, std::int64_t diagonal) {
    return where(triangular_mask(t, diagonal, true), t, zeros_like(t));
}

Tensor triu(const Tensor& t, std::int64_t diagonal) {
    return where(triangular_mask(t, diagonal, false), t, zeros_like(t));
}

} // namespace kael
