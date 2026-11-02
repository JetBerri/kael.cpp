#include "kael/autograd/gradcheck.h"

#include <cmath>
#include <random>
#include <sstream>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/ops.h"

#include "internal/index_checks.h"

namespace kael::autograd {

namespace {

Tensor random_projection(const Tensor& output, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> distribution;
    Tensor weights = empty(output.shape(), DType::F64);
    double* data = weights.data_ptr<double>();
    for (std::int64_t i = 0; i < weights.numel(); ++i) {
        data[i] = distribution(generator);
    }
    return weights;
}

double projected_value(const GradcheckFunction& function, const std::vector<Tensor>& inputs,
                       const Tensor& weights) {
    const NoGradGuard guard;
    return (function(inputs) * weights).sum().item();
}

double numeric_derivative(const GradcheckFunction& function, const std::vector<Tensor>& inputs,
                          double& element, const Tensor& weights, double epsilon) {
    const double original = element;
    element = original + epsilon;
    const double above = projected_value(function, inputs, weights);
    element = original - epsilon;
    const double below = projected_value(function, inputs, weights);
    element = original;
    return (above - below) / (2 * epsilon);
}

bool derivatives_close(double analytic, double numeric, const GradcheckOptions& options) {
    return std::abs(analytic - numeric) <=
           options.absolute_tolerance + options.relative_tolerance * std::abs(numeric);
}

void check_input(const Tensor& input, std::size_t position) {
    KAEL_CHECK_TYPE(input.dtype() == DType::F64,
                    "gradcheck input " << position << " must be F64, got " << input.dtype());
    KAEL_CHECK(input.requires_grad(), "gradcheck input " << position << " must require grad");
    KAEL_CHECK(input.is_leaf(), "gradcheck input " << position << " must be a leaf");
}

} // namespace

GradcheckResult gradcheck(const GradcheckFunction& function, const std::vector<Tensor>& inputs,
                          const GradcheckOptions& options) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        check_input(inputs[i], i);
    }
    const Tensor weights = random_projection(function(inputs), options.projection_seed);

    std::vector<Tensor> leaves = inputs;
    for (Tensor& input : leaves) {
        input.zero_grad();
    }
    (function(inputs) * weights).sum().backward();

    GradcheckResult result;
    std::ostringstream report;
    for (std::size_t position = 0; position < inputs.size(); ++position) {
        Tensor input = inputs[position];
        const std::optional<Tensor> gradient = input.grad();
        detail::for_each_index(input.shape(), [&](const Shape& index) {
            const double analytic = gradient ? gradient->at<double>(index) : 0.0;
            const double numeric = numeric_derivative(function, inputs, input.at<double>(index),
                                                      weights, options.epsilon);
            if (!derivatives_close(analytic, numeric, options)) {
                result.passed = false;
                report << "input " << position << " element " << to_string(index) << ": analytic "
                       << analytic << " != numeric " << numeric << "\n";
            }
        });
    }
    result.report = report.str();
    return result;
}

} // namespace kael::autograd
