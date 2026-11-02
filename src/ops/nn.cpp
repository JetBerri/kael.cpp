#include <algorithm>
#include <cmath>
#include <limits>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"
#include "kael/parallel.h"

#include "autograd/record.h"
#include "internal/nan_aware.h"

namespace kael {

namespace {

Tensor as_floating(const Tensor& t) {
    return is_floating_point(t.dtype()) ? t : t.to(DType::F32);
}

Tensor finite_or_zero(const Tensor& t) {
    const double infinity = std::numeric_limits<double>::infinity();
    return where(abs(t) == infinity, 0, t);
}

Tensor shifted_by_max(const Tensor& x, std::int64_t dim) {
    return x - max(x, {dim}, true);
}

constexpr std::int64_t kExponentialCost = 16;
constexpr std::int64_t kStandardizeCost = 4;

bool is_last_dim_of_contiguous(const Tensor& x, std::int64_t dim) {
    return x.dim() > 0 && x.is_contiguous() &&
           dim_position(dim, x.dim()) == static_cast<std::size_t>(x.dim() - 1);
}

template <typename T, typename RowKernel>
Tensor map_rows(const Tensor& x, std::int64_t row_size, std::int64_t cost, RowKernel kernel) {
    Tensor out = empty(x.shape(), x.dtype());
    const T* in = x.data_ptr<T>();
    T* results = out.data_ptr<T>();
    const std::int64_t grain = std::max<std::int64_t>(1, kElementGrain / (cost * row_size));
    parallel_for(0, x.numel() / row_size, grain, [&](std::int64_t first, std::int64_t last) {
        for (std::int64_t row = first; row < last; ++row) {
            kernel(in + row * row_size, results + row * row_size, row_size);
        }
    });
    return out;
}

template <typename T> T row_maximum(const T* row, std::int64_t size) {
    T largest = row[0];
    for (std::int64_t i = 1; i < size; ++i) {
        largest = detail::nan_aware_max(largest, row[i]);
    }
    return largest;
}

template <typename T> void softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = std::exp(row[i] - largest);
        total += static_cast<double>(out[i]);
    }
    const auto denominator = static_cast<T>(total);
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] /= denominator;
    }
}

template <typename T> void log_softmax_row(const T* row, T* out, std::int64_t size) {
    const T largest = row_maximum(row, size);
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = row[i] - largest;
        total += static_cast<double>(std::exp(out[i]));
    }
    const T log_total = std::log(static_cast<T>(total));
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] -= log_total;
    }
}

template <typename T> void standardize_row(const T* row, T* out, std::int64_t size, double eps) {
    double total = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        total += static_cast<double>(row[i]);
    }
    const double center = total / static_cast<double>(size);
    double squares = 0.0;
    for (std::int64_t i = 0; i < size; ++i) {
        const double deviation = static_cast<double>(row[i]) - center;
        squares += deviation * deviation;
    }
    const double scale = std::sqrt(squares / static_cast<double>(size) + eps);
    for (std::int64_t i = 0; i < size; ++i) {
        out[i] = static_cast<T>((static_cast<double>(row[i]) - center) / scale);
    }
}

} // namespace

Tensor logsumexp(const Tensor& t, std::int64_t dim, bool keepdim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return log(sum(x, {dim}, keepdim));
        }
        const Tensor largest = finite_or_zero(max(x, {dim}, true));
        const Tensor shifted_total = log(sum(exp(x - largest), {dim}, true)) + largest;
        return keepdim || x.dim() == 0 ? shifted_total : shifted_total.squeeze(dim);
    });
    autograd::record_logsumexp(out, t, dim, keepdim);
    return out;
}

Tensor softmax(const Tensor& t, std::int64_t dim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return x.clone();
        }
        if (is_last_dim_of_contiguous(x, dim)) {
            return dispatch_floating(x.dtype(), "softmax", [&]<typename T>() {
                return map_rows<T>(x, x.size(-1), kExponentialCost, softmax_row<T>);
            });
        }
        const Tensor exponentials = exp(shifted_by_max(x, dim));
        return exponentials / sum(exponentials, {dim}, true);
    });
    autograd::record_softmax(out, t, dim);
    return out;
}

Tensor log_softmax(const Tensor& t, std::int64_t dim) {
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t);
        if (x.numel() == 0) {
            return x.clone();
        }
        if (is_last_dim_of_contiguous(x, dim)) {
            return dispatch_floating(x.dtype(), "log_softmax", [&]<typename T>() {
                return map_rows<T>(x, x.size(-1), kExponentialCost, log_softmax_row<T>);
            });
        }
        const Tensor shifted = shifted_by_max(x, dim);
        return shifted - log(sum(exp(shifted), {dim}, true));
    });
    autograd::record_log_softmax(out, t, dim);
    return out;
}

Tensor standardize(const Tensor& t, std::int64_t normalized_dims, double eps) {
    KAEL_CHECK(normalized_dims > 0 && normalized_dims <= t.dim(),
               "standardize over " << normalized_dims << " dimensions got shape "
                                   << to_string(t.shape()));
    Tensor out = autograd::without_grad([&] {
        const Tensor x = as_floating(t).contiguous();
        if (x.numel() == 0) {
            return x.clone();
        }
        const auto trailing = x.shape().end() - static_cast<std::ptrdiff_t>(normalized_dims);
        const std::int64_t row_size = numel(Shape(trailing, x.shape().end()));
        return dispatch_floating(x.dtype(), "standardize", [&]<typename T>() {
            return map_rows<T>(x, row_size, kStandardizeCost,
                               [eps](const T* row, T* result, std::int64_t size) {
                                   standardize_row(row, result, size, eps);
                               });
        });
    });
    autograd::record_standardize(out, t, normalized_dims, eps);
    return out;
}

} // namespace kael
