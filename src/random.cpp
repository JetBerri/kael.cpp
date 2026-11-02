#include "kael/random.h"

#include <algorithm>
#include <cmath>

#include "kael/factory.h"

namespace kael {

namespace {
constexpr std::uint64_t kDefaultSeed = 0;
}

std::mt19937_64& global_generator() {
    // Function-local static: created on first use, which avoids the static
    // initialization order fiasco of a plain global.
    static std::mt19937_64 generator(kDefaultSeed);
    return generator;
}

void manual_seed(std::uint64_t seed) {
    global_generator().seed(seed);
}

namespace {

// Fills a new floating point tensor with samples drawn in double and
// converted to T, so F32 and F64 see the same sequence for the same seed.
template <typename Distribution>
Tensor sample_floating(const Shape& shape, DType dtype, const char* op, Distribution dist) {
    KAEL_CHECK(is_floating_point(dtype), op << " only supports floating point dtypes, got "
                                            << dtype);
    Tensor t = empty(shape, dtype);
    auto& gen = global_generator();
    dispatch_floating(dtype, op, [&]<typename T>() {
        T* out = t.data_ptr<T>();
        for (std::int64_t i = 0; i < t.numel(); ++i) {
            out[i] = static_cast<T>(dist(gen));
        }
    });
    return t;
}

} // namespace

Tensor rand(const Shape& shape, DType dtype) {
    Tensor t = sample_floating(shape, dtype, "rand", std::uniform_real_distribution<double>(0.0, 1.0));
    // Rounding a double just below 1 to float can produce exactly 1.0f;
    // keep the [0, 1) promise.
    if (dtype == DType::F32) {
        float* p = t.data_ptr<float>();
        for (std::int64_t i = 0; i < t.numel(); ++i) {
            if (p[i] >= 1.0f) {
                p[i] = std::nextafter(1.0f, 0.0f);
            }
        }
    }
    return t;
}

Tensor randn(const Shape& shape, DType dtype) {
    return sample_floating(shape, dtype, "randn", std::normal_distribution<double>(0.0, 1.0));
}

Tensor randint(std::int64_t low, std::int64_t high, const Shape& shape) {
    KAEL_CHECK(low < high, "randint needs low < high, got [" << low << ", " << high << ")");
    Tensor t = empty(shape, DType::I64);
    std::uniform_int_distribution<std::int64_t> dist(low, high - 1);
    auto& gen = global_generator();
    std::int64_t* out = t.data_ptr<std::int64_t>();
    for (std::int64_t i = 0; i < t.numel(); ++i) {
        out[i] = dist(gen);
    }
    return t;
}

Tensor randperm(std::int64_t n) {
    KAEL_CHECK(n >= 0, "randperm size must be non-negative, got " << n);
    Tensor t = arange(0.0, static_cast<double>(n), 1.0, DType::I64);
    std::int64_t* p = t.data_ptr<std::int64_t>();
    std::shuffle(p, p + n, global_generator());
    return t;
}

} // namespace kael
