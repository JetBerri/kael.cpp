#include <vector>

#include <gtest/gtest.h>

#include "kael/autograd/gradcheck.h"
#include "kael/autograd/node.h"
#include "kael/error.h"
#include "kael/kael.h"

using kael::Tensor;
using kael::autograd::gradcheck;
using Inputs = std::vector<Tensor>;

namespace {

Tensor input_with(const std::vector<double>& values, const kael::Shape& shape) {
    Tensor t = kael::from_vector(values, shape);
    t.requires_grad_();
    return t;
}

Tensor random_input(const kael::Shape& shape) {
    Tensor t = kael::randn(shape, kael::DType::F64);
    t.requires_grad_();
    return t;
}

Tensor positive_input(const kael::Shape& shape) {
    Tensor t = kael::rand(shape, kael::DType::F64) * 2.0 + 0.5;
    t.requires_grad_();
    return t;
}

Tensor away_from_zero(const kael::Shape& shape) {
    Tensor t = kael::randn(shape, kael::DType::F64);
    const kael::Tensor magnitudes = kael::abs(t) + 0.2;
    t = kael::where(t >= 0, magnitudes, -magnitudes);
    t.requires_grad_();
    return t;
}

#define EXPECT_GRADCHECK(function, ...)                                                            \
    do {                                                                                           \
        const auto gradcheck_result_ = gradcheck(function, __VA_ARGS__);                           \
        EXPECT_TRUE(gradcheck_result_.passed) << gradcheck_result_.report;                         \
    } while (0)

class AutogradFormulas : public testing::Test {
protected:
    void SetUp() override { kael::manual_seed(1234); }
};

} // namespace

TEST(Gradcheck, DetectsAWrongGradient) {
    const auto wrong = [](const Inputs& in) {
        const Tensor& x = in[0];
        const Tensor doubled = x * 2;
        return kael::autograd::GradMode::is_enabled() ? doubled : x * 3;
    };
    const auto result = gradcheck(wrong, {random_input({2})});
    EXPECT_FALSE(result.passed);
    EXPECT_NE(result.report.find("input 0 element [0]"), std::string::npos);
}

TEST(Gradcheck, RequiresF64Input) {
    Tensor single = kael::zeros({2});
    single.requires_grad_();
    EXPECT_THROW((void)gradcheck([](const Inputs& in) { return in[0]; }, {single}), kael::Error);
}

TEST(Gradcheck, RequiresInputWithGrad) {
    EXPECT_THROW((void)gradcheck([](const Inputs& in) { return in[0]; },
                                 {kael::zeros({2}, kael::DType::F64)}),
                 kael::Error);
}

TEST(Gradcheck, RestoresInputValues) {
    const Tensor x = input_with({1, 2}, {2});
    (void)gradcheck([](const Inputs& in) { return in[0] * in[0]; }, {x});
    EXPECT_EQ(x.at<double>({0}), 1.0);
    EXPECT_EQ(x.at<double>({1}), 2.0);
}

TEST_F(AutogradFormulas, Add) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] + in[1]; },
                     {random_input({2, 3}), random_input({2, 3})});
}

TEST_F(AutogradFormulas, AddBroadcast) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] + in[1]; },
                     {random_input({4, 3}), random_input({3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] + in[1]; },
                     {random_input({2, 1, 3}), random_input({4, 1})});
}

TEST_F(AutogradFormulas, Sub) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] - in[1]; },
                     {random_input({4, 3}), random_input({3})});
}

TEST_F(AutogradFormulas, Mul) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] * in[1]; },
                     {random_input({2, 3}), random_input({1, 3})});
}

TEST_F(AutogradFormulas, Div) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0] / in[1]; },
                     {random_input({2, 3}), positive_input({3})});
}

TEST_F(AutogradFormulas, PowBaseAndExponent) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::pow(in[0], in[1]); },
                     {positive_input({2, 3}), random_input({3})});
}

TEST_F(AutogradFormulas, PowScalarExponent) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::pow(in[0], 3); }, {random_input({3})});
}

TEST_F(AutogradFormulas, ScalarOperandsOnEitherSide) {
    EXPECT_GRADCHECK([](const Inputs& in) { return 2.5 * in[0] - 1.5 / (in[0] * in[0] + 1); },
                     {random_input({3})});
}

TEST_F(AutogradFormulas, MaximumAndMinimum) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::maximum(in[0], in[1]); },
                     {random_input({2, 3}), random_input({3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::minimum(in[0], in[1]); },
                     {random_input({2, 3}), random_input({3})});
}

TEST_F(AutogradFormulas, MixedPrecisionInputKeepsItsDtype) {
    Tensor single = kael::ones({2});
    single.requires_grad_();
    Tensor twice = kael::ones({2}, kael::DType::F64);
    twice.requires_grad_();
    (single * twice).sum().backward();
    EXPECT_EQ(single.grad()->dtype(), kael::DType::F32);
    EXPECT_EQ(twice.grad()->dtype(), kael::DType::F64);
}

TEST_F(AutogradFormulas, SmoothUnaries) {
    const std::vector<kael::autograd::GradcheckFunction> functions = {
        [](const Inputs& in) { return -in[0]; },
        [](const Inputs& in) { return kael::exp(in[0]); },
        [](const Inputs& in) { return kael::expm1(in[0]); },
        [](const Inputs& in) { return kael::sin(in[0]); },
        [](const Inputs& in) { return kael::cos(in[0]); },
        [](const Inputs& in) { return kael::tanh(in[0]); },
        [](const Inputs& in) { return kael::sigmoid(in[0]); },
        [](const Inputs& in) { return kael::gelu(in[0]); },
        [](const Inputs& in) { return kael::gelu(in[0], true); },
    };
    for (const auto& function : functions) {
        EXPECT_GRADCHECK(function, {random_input({2, 3})});
    }
}

TEST_F(AutogradFormulas, PositiveDomainUnaries) {
    const std::vector<kael::autograd::GradcheckFunction> functions = {
        [](const Inputs& in) { return kael::log(in[0]); },
        [](const Inputs& in) { return kael::log1p(in[0]); },
        [](const Inputs& in) { return kael::sqrt(in[0]); },
        [](const Inputs& in) { return kael::rsqrt(in[0]); },
    };
    for (const auto& function : functions) {
        EXPECT_GRADCHECK(function, {positive_input({2, 3})});
    }
}

TEST_F(AutogradFormulas, KinkedUnaries) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::relu(in[0]); }, {away_from_zero({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::abs(in[0]); }, {away_from_zero({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::clamp(in[0], -0.1, 0.1); },
                     {away_from_zero({2, 3})});
}

TEST_F(AutogradFormulas, ConventionsAtNonDifferentiablePoints) {
    const Tensor x = input_with({0}, {1});
    (kael::relu(x) + kael::abs(x)).sum().backward();
    EXPECT_EQ(x.grad()->at<double>({0}), 0.0);
}

TEST_F(AutogradFormulas, SumAndMean) {
    for (const bool keepdim : {false, true}) {
        const kael::autograd::GradcheckFunction sum_rows = [=](const Inputs& in) {
            return kael::sum(in[0], {1}, keepdim);
        };
        const kael::autograd::GradcheckFunction mean_rows = [=](const Inputs& in) {
            return kael::mean(in[0], {0, -1}, keepdim);
        };
        EXPECT_GRADCHECK(sum_rows, {random_input({2, 3, 2})});
        EXPECT_GRADCHECK(mean_rows, {random_input({2, 3, 2})});
    }
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::sum(in[0]); }, {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::mean(in[0]); }, {random_input({2, 3})});
}

TEST_F(AutogradFormulas, MeanExampleFromHand) {
    const Tensor x = input_with({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor y = kael::mean(x, {1});
    y.backward(kael::from_vector<double>({3, 6}));
    EXPECT_EQ(x.grad()->at<double>({0, 2}), 1.0);
    EXPECT_EQ(x.grad()->at<double>({1, 0}), 2.0);
}

TEST_F(AutogradFormulas, Prod) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::prod(in[0], {1}); },
                     {away_from_zero({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::prod(in[0]); }, {away_from_zero({2, 2})});
}

TEST_F(AutogradFormulas, ProdWithZeros) {
    const Tensor one_zero = input_with({2, 0, 3}, {3});
    kael::prod(one_zero).backward();
    EXPECT_EQ(one_zero.grad()->at<double>({0}), 0.0);
    EXPECT_EQ(one_zero.grad()->at<double>({1}), 6.0);
    EXPECT_EQ(one_zero.grad()->at<double>({2}), 0.0);

    const Tensor two_zeros = input_with({2, 0, 0}, {3});
    kael::prod(two_zeros).backward();
    EXPECT_EQ(two_zeros.grad()->at<double>({1}), 0.0);
}

TEST_F(AutogradFormulas, MaxAndMin) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::max(in[0], {1}); },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::min(in[0], {0}, true); },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::max(in[0]); }, {random_input({2, 3})});
}

TEST_F(AutogradFormulas, MaxSplitsTiesEvenly) {
    const Tensor x = input_with({3, 3, 1, 3}, {4});
    kael::max(x).backward();
    EXPECT_DOUBLE_EQ(x.grad()->at<double>({0}), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(x.grad()->at<double>({2}), 0.0);
}

TEST_F(AutogradFormulas, VarAndStd) {
    for (const std::int64_t correction : {0, 1}) {
        const kael::autograd::GradcheckFunction var_rows = [=](const Inputs& in) {
            return kael::var(in[0], {1}, false, correction);
        };
        const kael::autograd::GradcheckFunction std_all = [=](const Inputs& in) {
            return kael::std(in[0], {}, true, correction);
        };
        EXPECT_GRADCHECK(var_rows, {random_input({2, 3})});
        EXPECT_GRADCHECK(std_all, {random_input({2, 3})});
    }
}

TEST_F(AutogradFormulas, Norm) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::norm(in[0], {1}); },
                     {random_input({2, 3})});
    const Tensor zero = input_with({0, 0}, {2});
    kael::norm(zero).backward();
    EXPECT_EQ(zero.grad()->at<double>({0}), 0.0);
}

TEST_F(AutogradFormulas, Logsumexp) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::logsumexp(in[0], 1); },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::logsumexp(in[0], 0, true); },
                     {random_input({2, 3})});
}

TEST_F(AutogradFormulas, SoftmaxAndLogSoftmax) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::softmax(in[0], 1); },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::log_softmax(in[0], -1); },
                     {random_input({2, 3})});
}

TEST_F(AutogradFormulas, Views) {
    const std::vector<kael::autograd::GradcheckFunction> functions = {
        [](const Inputs& in) { return in[0].view({3, 2, 2}) * in[0].view({3, 2, 2}); },
        [](const Inputs& in) { return in[0].reshape({4, 3}); },
        [](const Inputs& in) { return in[0].transpose(0, 2) * 2; },
        [](const Inputs& in) { return in[0].permute({2, 0, 1}) * 2; },
        [](const Inputs& in) { return in[0].flatten() * 2; },
        [](const Inputs& in) { return in[0].slice(1, 0, 3, 2) * 3; },
        [](const Inputs& in) { return in[0].select(0, 1) * 3; },
        [](const Inputs& in) { return in[0].narrow(1, 1, 2) * 3; },
        [](const Inputs& in) { return in[0].transpose(0, 1).reshape({-1}) * 2; },
        [](const Inputs& in) { return in[0].unflatten(2, {1, 2}) * 2; },
        [](const Inputs& in) { return in[0].clone() * in[0]; },
        [](const Inputs& in) { return in[0].transpose(1, 2).contiguous() * 2; },
    };
    for (const auto& function : functions) {
        EXPECT_GRADCHECK(function, {random_input({2, 3, 2})});
    }
}

TEST_F(AutogradFormulas, SqueezeUnsqueezeExpand) {
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].squeeze() * 2; },
                     {random_input({2, 1, 3, 1})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].squeeze(1) * 2; },
                     {random_input({2, 1, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].unsqueeze(0) * 2; },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].expand({4, 2, 3}) * 2; },
                     {random_input({2, 1})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].T() * 2; }, {random_input({2, 3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return in[0].mT() * 2; }, {random_input({2, 2, 3})});
}

TEST_F(AutogradFormulas, SplitAccumulatesPieces) {
    EXPECT_GRADCHECK(
        [](const Inputs& in) {
            const auto pieces = in[0].split(2, 1);
            return kael::cat({pieces[1] * 2, pieces[0] * 3}, 1);
        },
        {random_input({2, 4})});
}

TEST_F(AutogradFormulas, NonContiguousLeaf) {
    const Tensor base = kael::randn({2, 3}, kael::DType::F64);
    Tensor strided = base.transpose(0, 1).detach();
    strided.requires_grad_();
    EXPECT_GRADCHECK(
        [](const Inputs& in) { return kael::exp(in[0]).sum({0}) * in[0].select(0, 0); }, {strided});
}

TEST_F(AutogradFormulas, Conversion) {
    EXPECT_GRADCHECK(
        [](const Inputs& in) { return in[0].to(kael::DType::F32).to(kael::DType::F64) * 2; },
        {random_input({2})},
        kael::autograd::GradcheckOptions{
            .epsilon = 1e-3, .absolute_tolerance = 1e-3, .relative_tolerance = 1e-2});
    const Tensor x = random_input({2});
    EXPECT_FALSE(x.to(kael::DType::I64).requires_grad());
}

TEST_F(AutogradFormulas, CatAndStack) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::cat({in[0], in[1]}, 1) * 2; },
                     {random_input({2, 3}), random_input({2, 1})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::stack({in[0], in[1]}, 1) * 2; },
                     {random_input({2, 3}), random_input({2, 3})});
}

TEST_F(AutogradFormulas, WhereAndMaskedFill) {
    const Tensor condition = kael::from_vector({true, false, false, true, true, false}, {2, 3});
    EXPECT_GRADCHECK([&](const Inputs& in) { return kael::where(condition, in[0], in[1]); },
                     {random_input({2, 3}), random_input({3})});
    EXPECT_GRADCHECK([&](const Inputs& in) { return kael::where(condition, in[0], 0.5); },
                     {random_input({2, 3})});
    EXPECT_GRADCHECK([&](const Inputs& in) { return kael::masked_fill(in[0], condition, 1.0); },
                     {random_input({1, 3})});
}

TEST_F(AutogradFormulas, IndexSelectAddsRepeatedIndices) {
    const Tensor indices = kael::from_vector<std::int64_t>({2, 0, 2});
    EXPECT_GRADCHECK([&](const Inputs& in) { return kael::index_select(in[0], 1, indices); },
                     {random_input({2, 3})});
    const Tensor x = input_with({1, 2, 3}, {3});
    kael::index_select(x, 0, indices).sum().backward();
    EXPECT_EQ(x.grad()->at<double>({2}), 2.0);
    EXPECT_EQ(x.grad()->at<double>({1}), 0.0);
}

TEST_F(AutogradFormulas, Gather) {
    const Tensor indices = kael::from_vector<std::int64_t>({0, 2, 1, 1}, {2, 2});
    EXPECT_GRADCHECK([&](const Inputs& in) { return kael::gather(in[0], 1, indices); },
                     {random_input({2, 3})});
}

TEST_F(AutogradFormulas, Matmul) {
    const auto multiply = [](const Inputs& in) { return kael::matmul(in[0], in[1]); };
    EXPECT_GRADCHECK(multiply, {random_input({2, 3}), random_input({3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({3}), random_input({3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({2, 3}), random_input({3})});
    EXPECT_GRADCHECK(multiply, {random_input({3}), random_input({3})});
    EXPECT_GRADCHECK(multiply, {random_input({4, 2, 3}), random_input({3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({2, 3}), random_input({4, 3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({2, 1, 2, 3}), random_input({2, 3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({3}), random_input({4, 3, 2})});
    EXPECT_GRADCHECK(multiply, {random_input({4, 2, 3}), random_input({3})});
}

TEST_F(AutogradFormulas, MatmulNamedForms) {
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::dot(in[0], in[1]); },
                     {random_input({3}), random_input({3})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::outer(in[0], in[1]); },
                     {random_input({3}), random_input({2})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::mm(in[0], in[1]); },
                     {random_input({2, 3}), random_input({3, 2})});
    EXPECT_GRADCHECK([](const Inputs& in) { return kael::bmm(in[0], in[1]); },
                     {random_input({2, 2, 3}), random_input({2, 3, 2})});
}

TEST_F(AutogradFormulas, MultiLayerComposition) {
    EXPECT_GRADCHECK(
        [](const Inputs& in) {
            const Tensor hidden = kael::tanh(kael::matmul(in[2], in[0].mT()) + in[1]);
            return kael::log_softmax(hidden, 1);
        },
        {random_input({2, 3}), random_input({2}), random_input({4, 3})});
}

TEST_F(AutogradFormulas, NonDifferentiableOpsHaveNoHistory) {
    const Tensor x = random_input({3});
    EXPECT_FALSE((x > 0).requires_grad());
    EXPECT_FALSE((x == x).requires_grad());
    EXPECT_FALSE(kael::argmax(x).requires_grad());
    EXPECT_FALSE(kael::logical_not(x).requires_grad());
    EXPECT_FALSE(kael::sign(x).requires_grad());
    EXPECT_FALSE(kael::floor_divide(x, 2).requires_grad());
    EXPECT_FALSE(kael::all(x).requires_grad());
}
