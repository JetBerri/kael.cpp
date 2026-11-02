#include "kael/random.h"

#include <algorithm>
#include <cmath>

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

} // namespace kael
