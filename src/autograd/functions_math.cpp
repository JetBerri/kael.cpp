#include <cmath>
#include <numbers>
#include <utility>

#include "kael/autograd/functions.h"
#include "kael/factory.h"
#include "kael/kernels.h"
#include "kael/ops.h"

#include "record.h"
#include "recording.h"

namespace kael::autograd {

namespace {

GradientFn matching_input(const Tensor& input, GradientFn gradient) {
    return [shape = input.shape(), dtype = input.dtype(), gradient = std::move(gradient)](
               const Tensor& g) { return sum_to_shape(gradient(g), shape).to(dtype); };
}

void binary_history(const Tensor& out, const char* name, const Tensor& a, const Tensor& b,
                    GradientFn gradient_a, GradientFn gradient_b) {
    set_history(
        out, name, {a, b},
        {matching_input(a, std::move(gradient_a)), matching_input(b, std::move(gradient_b))});
}

void unary_history(const Tensor& out, const char* name, const Tensor& x, GradientFn gradient) {
    set_history(out, name, {x}, {std::move(gradient)});
}

const auto kIdentity = [](const Tensor& g) { return g; };

template <typename T> T gelu_exact_slope(T x) {
    const T density = std::exp(-x * x / 2) / std::sqrt(2 * std::numbers::pi_v<T>);
    return T(0.5) * (1 + std::erf(x / std::numbers::sqrt2_v<T>)) + x * density;
}

template <typename T> T gelu_tanh_slope(T x) {
    const T scale = std::sqrt(2 / std::numbers::pi_v<T>);
    const T coefficient = T(0.044715);
    const T saturation = std::tanh(scale * (x + coefficient * x * x * x));
    return T(0.5) * (1 + saturation) +
           T(0.5) * x * (1 - saturation * saturation) * scale * (1 + 3 * coefficient * x * x);
}

Tensor extreme_weight(const Tensor& wins, const Tensor& ties, DType dtype) {
    return where(ties, kael::scalar(0.5, dtype), wins.to(dtype));
}

} // namespace

void record_add(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(out, "AddBackward", a, b, kIdentity, kIdentity);
    }
}

void record_sub(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(out, "SubBackward", a, b, kIdentity, [](const Tensor& g) { return -g; });
    }
}

void record_mul(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(
            out, "MulBackward", a, b,
            [saved = SavedTensor(b)](const Tensor& g) { return g * saved.get(); },
            [saved = SavedTensor(a)](const Tensor& g) { return g * saved.get(); });
    }
}

void record_div(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(
            out, "DivBackward", a, b,
            [divisor = SavedTensor(b)](const Tensor& g) { return g / divisor.get(); },
            [dividend = SavedTensor(a), divisor = SavedTensor(b)](const Tensor& g) {
                return -g * dividend.get() / (divisor.get() * divisor.get());
            });
    }
}

void record_pow(const Tensor& out, const Tensor& base, const Tensor& exponent) {
    if (should_record(out, {&base, &exponent})) {
        binary_history(
            out, "PowBackward", base, exponent,
            [x = SavedTensor(base), p = SavedTensor(exponent)](const Tensor& g) {
                return where(p.get() == 0, 0, g * p.get() * kael::pow(x.get(), p.get() - 1));
            },
            [x = SavedTensor(base), y = SavedTensor(out)](const Tensor& g) {
                return where(x.get() == 0, 0, g * y.get() * kael::log(x.get()));
            });
    }
}

void record_maximum(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(
            out, "MaximumBackward", a, b,
            [x = SavedTensor(a), y = SavedTensor(b)](const Tensor& g) {
                return g * extreme_weight(x.get() > y.get(), x.get() == y.get(), g.dtype());
            },
            [x = SavedTensor(a), y = SavedTensor(b)](const Tensor& g) {
                return g * extreme_weight(y.get() > x.get(), x.get() == y.get(), g.dtype());
            });
    }
}

void record_minimum(const Tensor& out, const Tensor& a, const Tensor& b) {
    if (should_record(out, {&a, &b})) {
        binary_history(
            out, "MinimumBackward", a, b,
            [x = SavedTensor(a), y = SavedTensor(b)](const Tensor& g) {
                return g * extreme_weight(x.get() < y.get(), x.get() == y.get(), g.dtype());
            },
            [x = SavedTensor(a), y = SavedTensor(b)](const Tensor& g) {
                return g * extreme_weight(y.get() < x.get(), x.get() == y.get(), g.dtype());
            });
    }
}

void record_neg(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "NegBackward", x, [](const Tensor& g) { return -g; });
    }
}

void record_abs(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "AbsBackward", x,
                      [saved = SavedTensor(x)](const Tensor& g) { return g * sign(saved.get()); });
    }
}

void record_exp(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "ExpBackward", x,
                      [y = SavedTensor(out)](const Tensor& g) { return g * y.get(); });
    }
}

void record_log(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "LogBackward", x,
                      [saved = SavedTensor(x)](const Tensor& g) { return g / saved.get(); });
    }
}

void record_log1p(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "Log1pBackward", x,
                      [saved = SavedTensor(x)](const Tensor& g) { return g / (saved.get() + 1); });
    }
}

void record_expm1(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "Expm1Backward", x,
                      [y = SavedTensor(out)](const Tensor& g) { return g * (y.get() + 1); });
    }
}

void record_sqrt(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "SqrtBackward", x,
                      [y = SavedTensor(out)](const Tensor& g) { return g / (y.get() * 2); });
    }
}

void record_rsqrt(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "RsqrtBackward", x,
                      [y = SavedTensor(out), saved = SavedTensor(x)](const Tensor& g) {
                          return g * y.get() / saved.get() * -0.5;
                      });
    }
}

void record_sin(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "SinBackward", x, [saved = SavedTensor(x)](const Tensor& g) {
            return g * kael::cos(saved.get());
        });
    }
}

void record_cos(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "CosBackward", x, [saved = SavedTensor(x)](const Tensor& g) {
            return -g * kael::sin(saved.get());
        });
    }
}

void record_tanh(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "TanhBackward", x, [y = SavedTensor(out)](const Tensor& g) {
            return g * (1 - y.get() * y.get());
        });
    }
}

void record_sigmoid(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "SigmoidBackward", x, [y = SavedTensor(out)](const Tensor& g) {
            return g * y.get() * (1 - y.get());
        });
    }
}

void record_relu(const Tensor& out, const Tensor& x) {
    if (should_record(out, {&x})) {
        unary_history(out, "ReluBackward", x,
                      [saved = SavedTensor(x)](const Tensor& g) { return g * (saved.get() > 0); });
    }
}

void record_gelu(const Tensor& out, const Tensor& x, bool approximate) {
    if (should_record(out, {&x})) {
        unary_history(out, "GeluBackward", x,
                      [saved = SavedTensor(x), approximate](const Tensor& g) {
                          const Tensor slope = floating_unary_kernel(saved.get(), [=](auto value) {
                              return approximate ? gelu_tanh_slope(value) : gelu_exact_slope(value);
                          });
                          return g * slope;
                      });
    }
}

void record_clamp(const Tensor& out, const Tensor& x, Scalar min, Scalar max) {
    if (should_record(out, {&x})) {
        unary_history(out, "ClampBackward", x, [saved = SavedTensor(x), min, max](const Tensor& g) {
            return g * logical_and(saved.get() >= min, saved.get() <= max);
        });
    }
}

} // namespace kael::autograd
