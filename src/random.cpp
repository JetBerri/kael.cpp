#include "kael/random.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "kael/factory.h"
#include "kael/ops.h"

namespace kael {

namespace {

constexpr std::uint64_t kDefaultSeed = 0;

template <typename Distribution>
Tensor sample_floating(const Shape& shape, DType dtype, const char* op, Distribution distribution) {
    Tensor t = empty(shape, dtype);
    dispatch_floating(dtype, op, [&]<typename T>() {
        std::generate_n(t.data_ptr<T>(), t.numel(),
                        [&] { return static_cast<T>(distribution(global_generator())); });
    });
    return t;
}

} // namespace

std::mt19937_64& global_generator() {
    static std::mt19937_64 generator(kDefaultSeed);
    return generator;
}

void manual_seed(std::uint64_t seed) {
    global_generator().seed(seed);
}

Tensor rand(const Shape& shape, DType dtype) {
    Tensor t =
        sample_floating(shape, dtype, "rand", std::uniform_real_distribution<double>(0.0, 1.0));
    if (dtype == DType::F32) {
        // A double just below 1 can round up to 1.0f, which is outside [0, 1).
        float* values = t.data_ptr<float>();
        std::replace(values, values + t.numel(), 1.0f, std::nextafter(1.0f, 0.0f));
    }
    return t;
}

Tensor randn(const Shape& shape, DType dtype) {
    return sample_floating(shape, dtype, "randn", std::normal_distribution<double>(0.0, 1.0));
}

Tensor randint(std::int64_t low, std::int64_t high, const Shape& shape) {
    KAEL_CHECK_VALUE(low < high, "randint needs low < high, got [" << low << ", " << high << ")");
    Tensor t = empty(shape, DType::I64);
    std::uniform_int_distribution<std::int64_t> distribution(low, high - 1);
    std::generate_n(t.data_ptr<std::int64_t>(), t.numel(),
                    [&] { return distribution(global_generator()); });
    return t;
}

Tensor randperm(std::int64_t n) {
    KAEL_CHECK_VALUE(n >= 0, "randperm size must be non-negative, got " << n);
    Tensor t = arange(0.0, static_cast<double>(n), 1.0, DType::I64);
    std::int64_t* values = t.data_ptr<std::int64_t>();
    std::shuffle(values, values + n, global_generator());
    return t;
}

Tensor bernoulli(const Shape& shape, double probability, DType dtype) {
    KAEL_CHECK_VALUE(probability >= 0.0 && probability <= 1.0,
                     "bernoulli probability must be in [0, 1], got " << probability);
    return (rand(shape, DType::F64) < probability).to(dtype);
}

Tensor multinomial(const Tensor& weights, std::int64_t samples, bool replacement) {
    KAEL_CHECK_TYPE(is_floating_point(weights.dtype()),
                    "multinomial needs floating point weights, got " << weights.dtype());
    KAEL_CHECK_VALUE(weights.dim() == 1 || weights.dim() == 2,
                     "multinomial needs weights of shape [K] or [B, K], got "
                         << to_string(weights.shape()));
    KAEL_CHECK_VALUE(samples > 0, "multinomial needs a positive sample count, got " << samples);

    KAEL_CHECK_VALUE(weights.numel() > 0,
                     "multinomial needs weights, got shape " << to_string(weights.shape()));

    const Tensor rows = weights.detach().cpu().to(DType::F64).contiguous();
    const std::int64_t classes = rows.size(-1);
    const std::int64_t row_count = rows.numel() / classes;
    Shape out_shape = weights.shape();
    out_shape.back() = samples;
    Tensor out = empty(out_shape, DType::I64);
    auto* results = out.data_ptr<std::int64_t>();
    const double* data = rows.data_ptr<double>();

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> remaining(static_cast<std::size_t>(classes));
    std::vector<double> cumulative(static_cast<std::size_t>(classes));
    for (std::int64_t row = 0; row < row_count; ++row) {
        const double* row_weights = data + row * classes;
        std::copy_n(row_weights, classes, remaining.begin());
        std::int64_t positive = 0;
        for (const double weight : remaining) {
            KAEL_CHECK_VALUE(std::isfinite(weight) && weight >= 0.0,
                             "multinomial weights must be finite and non-negative, got " << weight);
            positive += weight > 0.0 ? 1 : 0;
        }
        KAEL_CHECK_VALUE(positive > 0, "multinomial needs at least one positive weight");
        KAEL_CHECK_VALUE(replacement || samples <= positive,
                         "cannot draw " << samples << " samples without replacement from "
                                        << positive << " positive weights");
        for (std::int64_t sample = 0; sample < samples; ++sample) {
            std::partial_sum(remaining.begin(), remaining.end(), cumulative.begin());
            const double point = uniform(global_generator()) * cumulative.back();
            auto found = std::upper_bound(cumulative.begin(), cumulative.end(), point);
            found = found == cumulative.end() ? found - 1 : found;
            auto picked = found - cumulative.begin();
            while (remaining[static_cast<std::size_t>(picked)] <= 0.0 && picked > 0) {
                --picked;
            }
            results[row * samples + sample] = picked;
            if (!replacement) {
                remaining[static_cast<std::size_t>(picked)] = 0.0;
            }
        }
    }
    return out.to(weights.device());
}

} // namespace kael
