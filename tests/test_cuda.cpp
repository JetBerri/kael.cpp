#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "kael/autograd/gradcheck.h"
#include "kael/kael.h"
#include "kael/nn/activation.h"
#include "kael/nn/embedding.h"
#include "kael/nn/layernorm.h"
#include "kael/nn/linear.h"
#include "kael/nn/loss.h"
#include "kael/nn/sequential.h"
#include "kael/nn/transformer.h"
#include "kael/optim/adam.h"
#include "kael/optim/clip.h"

#include "support/allclose.h"

using kael::Device;
using kael::DType;
using kael::Tensor;

namespace {

class Cuda : public testing::Test {
protected:
    void SetUp() override {
        if (!kael::cuda_is_available()) {
            GTEST_SKIP() << "no CUDA device";
        }
        kael::manual_seed(2024);
    }
};

kael::test::Tolerance tolerance_for(DType dtype) {
    return dtype == DType::F64 ? kael::test::Tolerance{1e-10, 1e-10}
                               : kael::test::Tolerance{1e-4, 1e-4};
}

void expect_matches_cpu(const Tensor& on_device, const Tensor& on_cpu, const std::string& name) {
    EXPECT_EQ(on_device.device(), Device::CUDA) << name;
    EXPECT_EQ(on_device.shape(), on_cpu.shape()) << name;
    EXPECT_EQ(on_device.dtype(), on_cpu.dtype()) << name;
    EXPECT_TRUE(kael::test::allclose(
        on_device.cpu(), on_cpu, {.equal_nan = true, .tolerance = tolerance_for(on_cpu.dtype())}))
        << name;
}

using UnaryOp = std::pair<std::string, std::function<Tensor(const Tensor&)>>;
using BinaryOp = std::pair<std::string, std::function<Tensor(const Tensor&, const Tensor&)>>;

std::vector<UnaryOp> unary_ops() {
    return {
        {"neg", [](const Tensor& x) { return kael::neg(x); }},
        {"abs", [](const Tensor& x) { return kael::abs(x); }},
        {"sign", [](const Tensor& x) { return kael::sign(x); }},
        {"exp", [](const Tensor& x) { return kael::exp(x); }},
        {"log", [](const Tensor& x) { return kael::log(kael::abs(x) + 1); }},
        {"log1p", [](const Tensor& x) { return kael::log1p(kael::abs(x)); }},
        {"expm1", [](const Tensor& x) { return kael::expm1(x); }},
        {"sqrt", [](const Tensor& x) { return kael::sqrt(kael::abs(x)); }},
        {"rsqrt", [](const Tensor& x) { return kael::rsqrt(kael::abs(x) + 1); }},
        {"sin", [](const Tensor& x) { return kael::sin(x); }},
        {"cos", [](const Tensor& x) { return kael::cos(x); }},
        {"tanh", [](const Tensor& x) { return kael::tanh(x); }},
        {"sigmoid", [](const Tensor& x) { return kael::sigmoid(x); }},
        {"relu", [](const Tensor& x) { return kael::relu(x); }},
        {"gelu", [](const Tensor& x) { return kael::gelu(x); }},
        {"gelu_tanh", [](const Tensor& x) { return kael::gelu(x, true); }},
        {"clamp", [](const Tensor& x) { return kael::clamp(x, -0.5, 0.5); }},
        {"clone", [](const Tensor& x) { return x.clone(); }},
        {"to_f64", [](const Tensor& x) { return x.to(DType::F64); }},
        {"to_i64", [](const Tensor& x) { return x.to(DType::I64); }},
        {"to_bool", [](const Tensor& x) { return x.to(DType::Bool); }},
        {"sum", [](const Tensor& x) { return kael::sum(x); }},
        {"sum_dim", [](const Tensor& x) { return kael::sum(x, {0}); }},
        {"sum_keep", [](const Tensor& x) { return kael::sum(x, {-1}, true); }},
        {"mean", [](const Tensor& x) { return kael::mean(x); }},
        {"mean_dim", [](const Tensor& x) { return kael::mean(x, {1}); }},
        {"prod", [](const Tensor& x) { return kael::prod(x, {1}); }},
        {"max", [](const Tensor& x) { return kael::max(x); }},
        {"max_dim", [](const Tensor& x) { return kael::max(x, {0}); }},
        {"min_dim", [](const Tensor& x) { return kael::min(x, {1}, true); }},
        {"argmax", [](const Tensor& x) { return kael::argmax(x, 1); }},
        {"argmin", [](const Tensor& x) { return kael::argmin(x); }},
        {"var", [](const Tensor& x) { return kael::var(x, {1}); }},
        {"std", [](const Tensor& x) { return kael::std(x); }},
        {"norm", [](const Tensor& x) { return kael::norm(x, {0}); }},
        {"all", [](const Tensor& x) { return kael::all(x, {0}); }},
        {"any", [](const Tensor& x) { return kael::any(x); }},
        {"softmax_last", [](const Tensor& x) { return kael::softmax(x, -1); }},
        {"softmax_first", [](const Tensor& x) { return kael::softmax(x, 0); }},
        {"log_softmax", [](const Tensor& x) { return kael::log_softmax(x, -1); }},
        {"logsumexp", [](const Tensor& x) { return kael::logsumexp(x, 1); }},
        {"standardize", [](const Tensor& x) { return kael::standardize(x, 1, 1e-5); }},
        {"logical_not", [](const Tensor& x) { return kael::logical_not(x); }},
    };
}

std::vector<BinaryOp> binary_ops() {
    return {
        {"add", [](const Tensor& a, const Tensor& b) { return a + b; }},
        {"sub", [](const Tensor& a, const Tensor& b) { return a - b; }},
        {"mul", [](const Tensor& a, const Tensor& b) { return a * b; }},
        {"div", [](const Tensor& a, const Tensor& b) { return a / (kael::abs(b) + 1); }},
        {"floor_divide",
         [](const Tensor& a, const Tensor& b) { return kael::floor_divide(a, kael::abs(b) + 1); }},
        {"pow", [](const Tensor& a, const Tensor& b) { return kael::pow(kael::abs(a), b); }},
        {"maximum", [](const Tensor& a, const Tensor& b) { return kael::maximum(a, b); }},
        {"minimum", [](const Tensor& a, const Tensor& b) { return kael::minimum(a, b); }},
        {"eq", [](const Tensor& a, const Tensor& b) { return a == b; }},
        {"ne", [](const Tensor& a, const Tensor& b) { return a != b; }},
        {"lt", [](const Tensor& a, const Tensor& b) { return a < b; }},
        {"le", [](const Tensor& a, const Tensor& b) { return a <= b; }},
        {"gt", [](const Tensor& a, const Tensor& b) { return a > b; }},
        {"ge", [](const Tensor& a, const Tensor& b) { return a >= b; }},
        {"logical_and", [](const Tensor& a, const Tensor& b) { return kael::logical_and(a, b); }},
        {"logical_or", [](const Tensor& a, const Tensor& b) { return kael::logical_or(a, b); }},
        {"logical_xor", [](const Tensor& a, const Tensor& b) { return kael::logical_xor(a, b); }},
        {"where", [](const Tensor& a, const Tensor& b) { return kael::where(a > 0, a, b); }},
        {"masked_fill",
         [](const Tensor& a, const Tensor& b) { return kael::masked_fill(a, b > 0, 7.0); }},
    };
}

} // namespace

TEST_F(Cuda, TransfersRoundTrip) {
    for (const DType dtype : {DType::F32, DType::F64, DType::I64, DType::Bool}) {
        const Tensor x = kael::randn({5, 4}).to(dtype);
        const Tensor on_device = x.cuda();
        EXPECT_EQ(on_device.device(), Device::CUDA);
        EXPECT_EQ(on_device.dtype(), dtype);
        EXPECT_FALSE(on_device.shares_storage(x));
        const Tensor back = on_device.cpu();
        EXPECT_EQ(back.device(), Device::CPU);
        EXPECT_TRUE(kael::equal(back, x));
    }
}

TEST_F(Cuda, SameDeviceReturnsTheSameTensor) {
    const Tensor x = kael::ones({2}).cuda();
    EXPECT_TRUE(x.cuda().is_same(x));
    const Tensor y = kael::ones({2});
    EXPECT_TRUE(y.cpu().is_same(y));
}

TEST_F(Cuda, NonContiguousTensorsTransfer) {
    const Tensor x = kael::randn({4, 6});
    const Tensor view = x.T().slice(0, 1, 6, 2);
    EXPECT_FALSE(view.is_contiguous());
    EXPECT_TRUE(kael::equal(view.cuda().cpu(), view.contiguous()));
}

TEST_F(Cuda, FactoriesCreateOnTheDevice) {
    const Tensor z = kael::zeros({3, 2}, DType::F64, Device::CUDA);
    EXPECT_EQ(z.device(), Device::CUDA);
    EXPECT_TRUE(kael::equal(z.cpu(), kael::zeros({3, 2}, DType::F64)));
    EXPECT_TRUE(
        kael::equal(kael::ones({2}, DType::I64, Device::CUDA).cpu(), kael::ones({2}, DType::I64)));
    EXPECT_TRUE(kael::equal(kael::full({2}, 2.5, DType::F32, Device::CUDA).cpu(),
                            kael::full({2}, 2.5, DType::F32)));
    EXPECT_EQ(kael::zeros_like(z).device(), Device::CUDA);
    EXPECT_EQ(kael::ones_like(z).device(), Device::CUDA);
    EXPECT_EQ(kael::empty_like(z).device(), Device::CUDA);
}

TEST_F(Cuda, HostOnlyAccessFailsClearly) {
    Tensor x = kael::ones({2}).cuda();
    EXPECT_THROW((void)x.data_ptr<float>(), kael::Error);
    EXPECT_THROW((void)x.at<float>({0}), kael::Error);
    EXPECT_NO_THROW((void)x.device_data());
}

TEST_F(Cuda, ReadingValuesCopiesToHost) {
    const Tensor x = kael::full({1}, 3.5, DType::F64).cuda();
    EXPECT_EQ(x.item<double>(), 3.5);
    EXPECT_EQ(x.item(), 3.5);
    std::ostringstream text;
    text << kael::ones({2}).cuda();
    EXPECT_NE(text.str().find("1.0000"), std::string::npos);
    EXPECT_NE(text.str().find("device=cuda"), std::string::npos);
}

TEST_F(Cuda, MixedDevicesAreRejected) {
    const Tensor on_device = kael::ones({2}).cuda();
    const Tensor on_cpu = kael::ones({2});
    try {
        (void)(on_device + on_cpu);
        FAIL() << "should have thrown";
    } catch (const kael::Error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("same device"), std::string::npos) << message;
    }
    EXPECT_THROW((void)kael::matmul(kael::ones({2, 2}).cuda(), kael::ones({2, 2})), kael::Error);
    EXPECT_THROW((void)kael::cat({on_device, on_cpu}), kael::Error);
    EXPECT_THROW((void)kael::gather(on_cpu, 0, kael::zeros({1}, DType::I64).cuda()), kael::Error);
}

TEST_F(Cuda, ScalarsFollowTheTensorDevice) {
    const Tensor x = kael::full({2}, 2.0).cuda();
    expect_matches_cpu(x * 3.0, kael::full({2}, 6.0), "tensor * scalar");
    expect_matches_cpu(3.0 - x, kael::full({2}, 1.0), "scalar - tensor");
    Tensor y = x.clone();
    y += 1;
    expect_matches_cpu(y, kael::full({2}, 3.0), "in-place scalar");
}

TEST_F(Cuda, CopyBetweenDevicesWorks) {
    Tensor target = kael::zeros({2, 2}).cuda();
    const Tensor source = kael::full({2, 2}, 4.0);
    target.copy_(source);
    expect_matches_cpu(target, source, "cpu to device");
    Tensor host = kael::zeros({2, 2});
    host.copy_(target);
    EXPECT_TRUE(kael::equal(host, source));
}

TEST_F(Cuda, UnaryOpsMatchCpuOnEveryLayout) {
    const Tensor base = kael::randn({6, 8});
    const std::vector<std::pair<std::string, Tensor>> layouts = {
        {"contiguous", base},
        {"transposed", base.T()},
        {"strided", base.slice(0, 0, 6, 2).slice(1, 1, 8, 2)},
    };
    for (const auto& [layout, input] : layouts) {
        const Tensor on_device = input.cuda();
        for (const auto& [name, op] : unary_ops()) {
            expect_matches_cpu(op(on_device), op(input), name + "/" + layout);
        }
        const Tensor device_view = base.cuda();
        if (layout == "transposed") {
            for (const auto& [name, op] : unary_ops()) {
                expect_matches_cpu(op(device_view.T()), op(input), name + "/device view");
            }
        }
    }
}

TEST_F(Cuda, DoublePrecisionMatchesCpu) {
    const Tensor x = kael::randn({6, 8}, DType::F64);
    for (const auto& [name, op] : unary_ops()) {
        expect_matches_cpu(op(x.cuda()), op(x), name + "/f64");
    }
}

TEST_F(Cuda, BinaryOpsMatchCpuWithBroadcast) {
    const Tensor a = kael::randn({6, 8});
    const std::vector<std::pair<std::string, Tensor>> others = {
        {"same", kael::randn({6, 8})},
        {"row", kael::randn({8})},
        {"column", kael::randn({6, 1})},
        {"zero_dim", kael::randn({})},
    };
    for (const auto& [shape_name, b] : others) {
        for (const auto& [name, op] : binary_ops()) {
            expect_matches_cpu(op(a.cuda(), b.cuda()), op(a, b), name + "/" + shape_name);
        }
    }
}

TEST_F(Cuda, IntegerOpsMatchCpu) {
    const Tensor a = kael::randint(-20, 20, {5, 5});
    const Tensor b = kael::randint(1, 9, {5, 5});
    expect_matches_cpu(a.cuda() + b.cuda(), a + b, "add i64");
    expect_matches_cpu(a.cuda() * b.cuda(), a * b, "mul i64");
    expect_matches_cpu(kael::pow(a.cuda(), b.cuda()), kael::pow(a, b), "pow i64");
    expect_matches_cpu(kael::floor_divide(a.cuda(), b.cuda()), kael::floor_divide(a, b),
                       "floor_divide i64");
    expect_matches_cpu(kael::sum(a.cuda()), kael::sum(a), "sum i64");
    expect_matches_cpu(kael::max(a.cuda(), {0}), kael::max(a, {0}), "max i64");
    expect_matches_cpu(kael::abs(a.cuda()), kael::abs(a), "abs i64");
    expect_matches_cpu(a.cuda() / b.cuda(), a / b, "div i64");
}

TEST_F(Cuda, BoolReductionsMatchCpu) {
    const Tensor a = kael::randn({5, 5}) > 0;
    expect_matches_cpu(kael::sum(a.cuda()), kael::sum(a), "sum bool");
    expect_matches_cpu(kael::sum(a.cuda(), {1}), kael::sum(a, {1}), "sum dim bool");
}

TEST_F(Cuda, NonFiniteValuesMatchCpu) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Tensor x =
        kael::from_vector<float>({1.0f, static_cast<float>(inf), 2.0f, static_cast<float>(-inf),
                                  1.0f, 2.0f, static_cast<float>(nan), 1.0f, 2.0f},
                                 {3, 3});
    expect_matches_cpu(kael::max(x.cuda(), {1}), kael::max(x, {1}), "max");
    expect_matches_cpu(kael::min(x.cuda(), {1}), kael::min(x, {1}), "min");
    expect_matches_cpu(kael::sum(x.cuda(), {1}), kael::sum(x, {1}), "sum");
    expect_matches_cpu(kael::softmax(x.cuda(), -1), kael::softmax(x, -1), "softmax");
    expect_matches_cpu(kael::maximum(x.cuda(), x.T().cuda()), kael::maximum(x, x.T()), "maximum");
    expect_matches_cpu(kael::clamp(x.cuda(), 0.0, 1.0), kael::clamp(x, 0.0, 1.0), "clamp");
}

TEST_F(Cuda, LongSumsMatchCpuAndAreDeterministic) {
    const Tensor x = kael::randn({200000}, DType::F64);
    const Tensor first = kael::sum(x.cuda());
    const Tensor second = kael::sum(x.cuda());
    EXPECT_TRUE(kael::equal(first.cpu(), second.cpu()));
    expect_matches_cpu(first, kael::sum(x), "long sum");
    const Tensor wide = kael::randn({3, 100000});
    expect_matches_cpu(kael::sum(wide.cuda(), {1}), kael::sum(wide, {1}), "wide rows");
    expect_matches_cpu(kael::max(wide.cuda(), {1}), kael::max(wide, {1}), "wide max");
}

TEST_F(Cuda, EmptyTensorsWork) {
    const Tensor empty = kael::zeros({0, 3});
    expect_matches_cpu(empty.cuda() + empty.cuda(), empty + empty, "add");
    expect_matches_cpu(kael::sum(empty.cuda()), kael::sum(empty), "sum");
    expect_matches_cpu(kael::sum(empty.cuda(), {0}), kael::sum(empty, {0}), "sum dim");
    expect_matches_cpu(kael::prod(empty.cuda()), kael::prod(empty), "prod");
    EXPECT_THROW((void)kael::max(empty.cuda()), kael::Error);
}

class CudaMatmul : public testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(CudaMatmul, MatchesCpu) {
    if (!kael::cuda_is_available()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const auto [rows, inner, columns] = GetParam();
    kael::manual_seed(77);
    for (const DType dtype : {DType::F32, DType::F64}) {
        const Tensor a = kael::randn({rows, inner}, dtype);
        const Tensor b = kael::randn({inner, columns}, dtype);
        expect_matches_cpu(kael::matmul(a.cuda(), b.cuda()), kael::matmul(a, b), "matmul");
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CudaMatmul,
                         testing::Values(std::tuple{1, 1, 1}, std::tuple{4, 4, 4},
                                         std::tuple{5, 3, 9}, std::tuple{8, 16, 7},
                                         std::tuple{33, 20, 65}, std::tuple{64, 1, 64},
                                         std::tuple{1, 64, 1}));

TEST_F(Cuda, MatmulBatchesVectorsAndIntegersMatchCpu) {
    const Tensor a = kael::randn({3, 1, 4, 5});
    const Tensor b = kael::randn({2, 5, 6});
    expect_matches_cpu(kael::matmul(a.cuda(), b.cuda()), kael::matmul(a, b), "batched");
    const Tensor v = kael::randn({5});
    expect_matches_cpu(kael::matmul(a.cuda(), v.cuda()), kael::matmul(a, v), "matrix-vector");
    expect_matches_cpu(kael::dot(v.cuda(), v.cuda()), kael::dot(v, v), "dot");
    expect_matches_cpu(kael::outer(v.cuda(), v.cuda()), kael::outer(v, v), "outer");
    const Tensor i = kael::randint(-5, 5, {6, 6});
    expect_matches_cpu(kael::matmul(i.cuda(), i.cuda()), kael::matmul(i, i), "i64");
    const Tensor t = kael::randn({5, 4});
    expect_matches_cpu(kael::matmul(t.T().cuda(), t.cuda()), kael::matmul(t.T(), t), "transposed");
}

TEST_F(Cuda, IndexingMatchesCpu) {
    const Tensor x = kael::randn({6, 5});
    const Tensor rows = kael::from_vector<std::int64_t>({5, 0, 0, 2});
    const Tensor picks = kael::randint(0, 5, {6, 3});
    const Tensor device_x = x.cuda();
    expect_matches_cpu(kael::index_select(device_x, 0, rows.cuda()), kael::index_select(x, 0, rows),
                       "index_select");
    expect_matches_cpu(kael::gather(device_x, 1, picks.cuda()), kael::gather(x, 1, picks),
                       "gather");
    expect_matches_cpu(kael::gather(device_x.T(), 0, picks.T().cuda()),
                       kael::gather(x.T(), 0, picks.T()), "gather transposed");
    expect_matches_cpu(kael::cat({device_x, device_x}, 1), kael::cat({x, x}, 1), "cat");
    expect_matches_cpu(kael::stack({device_x, device_x}), kael::stack({x, x}), "stack");

    Tensor grid = kael::zeros({6, 5});
    Tensor device_grid = grid.clone().cuda();
    const Tensor source = kael::randn({6, 3});
    grid.scatter_add_(1, picks, source);
    device_grid.scatter_add_(1, picks.cuda(), source.cuda());
    expect_matches_cpu(device_grid, grid, "scatter_add_");

    Tensor added = kael::zeros({6, 5});
    Tensor device_added = added.clone().cuda();
    const Tensor rows_source = kael::randn({4, 5});
    added.index_add_(0, rows, rows_source);
    device_added.index_add_(0, rows.cuda(), rows_source.cuda());
    expect_matches_cpu(device_added, added, "index_add_");
}

TEST_F(Cuda, OutOfRangeIndicesStillThrow) {
    const Tensor x = kael::zeros({2, 2}).cuda();
    const Tensor bad = kael::full({1}, 9, DType::I64).cuda();
    EXPECT_THROW((void)kael::index_select(x, 0, bad), kael::IndexError);
    EXPECT_THROW((void)kael::gather(x, 0, kael::full({1, 2}, 9, DType::I64).cuda()),
                 kael::IndexError);
    EXPECT_THROW((void)kael::gather(x, 0, kael::full({1, 2}, -1, DType::I64).cuda()),
                 kael::IndexError);
}

TEST_F(Cuda, BackwardMatchesCpuAndStaysOnTheDevice) {
    const Tensor x = kael::randn({4, 3}, DType::F64).requires_grad_();
    const Tensor device_x = x.detach().clone().cuda().requires_grad_();
    const auto loss = [](const Tensor& t) {
        const Tensor hidden = kael::tanh(kael::matmul(t, t.T()));
        return kael::sum(kael::softmax(hidden, -1) * hidden) + kael::mean(kael::exp(t)) +
               kael::sum(kael::gelu(t, true)) + kael::sum(kael::max(t, {1}));
    };
    loss(x).backward();
    loss(device_x).backward();
    EXPECT_EQ(device_x.grad()->device(), Device::CUDA);
    expect_matches_cpu(*device_x.grad(), *x.grad(), "gradient");
}

TEST_F(Cuda, GradientsCrossDevicesBackToTheOrigin) {
    const Tensor x = kael::randn({3}, DType::F64).requires_grad_();
    const Tensor on_device = x.cuda();
    kael::sum(on_device * on_device).backward();
    EXPECT_EQ(x.grad()->device(), Device::CPU);
    EXPECT_TRUE(kael::test::allclose(*x.grad(), x.detach() * 2));

    const Tensor y = kael::randn({3}, DType::F64).cuda().requires_grad_();
    kael::sum(y.cpu() * 3).backward();
    EXPECT_EQ(y.grad()->device(), Device::CUDA);
}

TEST_F(Cuda, ViewAndIndexingGradientsMatchCpu) {
    const Tensor x = kael::randn({4, 5}, DType::F64).requires_grad_();
    const Tensor device_x = x.detach().clone().cuda().requires_grad_();
    const Tensor rows = kael::from_vector<std::int64_t>({1, 1, 3});
    const auto loss = [](const Tensor& t, const Tensor& picked) {
        const Tensor narrowed = t.slice(1, 1, 5, 2).select(0, 2);
        const Tensor selected = kael::index_select(t, 0, picked);
        const Tensor gathered = kael::gather(t, 1, kael::zeros_like(t).to(kael::DType::I64));
        return kael::sum(narrowed * narrowed) + kael::sum(selected * 2) + kael::sum(gathered) +
               kael::sum(kael::cat({t, t}, 0).T()) + kael::sum(kael::log_softmax(t, 0));
    };
    loss(x, rows).backward();
    loss(device_x, rows.cuda()).backward();
    expect_matches_cpu(*device_x.grad(), *x.grad(), "gradient");
}

TEST_F(Cuda, GradcheckRunsOnTheDevice) {
    const Tensor x = kael::randn({3, 2}, DType::F64).requires_grad_();
    const auto result = kael::autograd::gradcheck(
        [](const std::vector<Tensor>& in) {
            return kael::sum(kael::tanh(in[0].cuda()) * in[0].cuda()).cpu();
        },
        {x});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST_F(Cuda, ModuleToKeepsParameterIdentityAndValues) {
    kael::nn::Linear layer(3, 2);
    const Tensor weight = layer.weight();
    const Tensor before = weight.detach().clone();
    layer.to(Device::CUDA);
    EXPECT_TRUE(layer.weight().is_same(weight));
    EXPECT_EQ(weight.device(), Device::CUDA);
    EXPECT_TRUE(weight.requires_grad());
    EXPECT_TRUE(kael::equal(weight.detach().cpu(), before));
    layer.to(Device::CPU);
    EXPECT_EQ(weight.device(), Device::CPU);
    EXPECT_TRUE(kael::equal(weight.detach(), before));
}

TEST_F(Cuda, ModuleToMovesExistingGradients) {
    kael::nn::Linear layer(2, 1);
    kael::sum(layer(kael::ones({1, 2}))).backward();
    layer.to(Device::CUDA);
    ASSERT_TRUE(layer.weight().grad().has_value());
    EXPECT_EQ(layer.weight().grad()->device(), Device::CUDA);
}

TEST_F(Cuda, IntegerClampAndMaskedFillKeepEveryBit) {
    const std::int64_t large = (std::int64_t{1} << 53) + 1;
    const Tensor values =
        kael::from_vector(std::vector<std::int64_t>{0, large + 10}).to(Device::CUDA);
    const Tensor clamped = kael::clamp(values, large, large);
    EXPECT_EQ(clamped.cpu().data_ptr<std::int64_t>()[0], large);
    EXPECT_EQ(clamped.cpu().data_ptr<std::int64_t>()[1], large);
    const Tensor mask = kael::from_vector({true, false}).to(Device::CUDA);
    const Tensor filled = kael::masked_fill(values, mask, large);
    EXPECT_EQ(filled.cpu().data_ptr<std::int64_t>()[0], large);
}

TEST_F(Cuda, IntegerSumWrapsAndFloatConversionSaturates) {
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
    const Tensor longs = kael::from_vector(std::vector<std::int64_t>{kMax, 1}).to(Device::CUDA);
    EXPECT_EQ(kael::sum(longs).item<std::int64_t>(), std::numeric_limits<std::int64_t>::min());
    const Tensor floats =
        kael::from_vector(std::vector<double>{std::nan(""), 1e300}).to(Device::CUDA);
    const Tensor converted = floats.to(DType::I64).cpu();
    EXPECT_EQ(converted.data_ptr<std::int64_t>()[0], 0);
    EXPECT_EQ(converted.data_ptr<std::int64_t>()[1], kMax);
}

TEST_F(Cuda, LayersMatchCpu) {
    kael::nn::LayerNorm norm({4}, 1e-5, DType::F64);
    const Tensor x = kael::randn({2, 3, 4}, DType::F64);
    const Tensor expected = norm(x);
    norm.to(Device::CUDA);
    expect_matches_cpu(norm(x.cuda()), expected, "layer norm");

    kael::nn::Embedding embedding(10, 4);
    const Tensor tokens = kael::randint(0, 10, {2, 3});
    const Tensor embedded = embedding(tokens);
    embedding.to(Device::CUDA);
    expect_matches_cpu(embedding(tokens.cuda()), embedded, "embedding");
}

TEST_F(Cuda, TrainingMatchesCpuStepByStep) {
    const auto build = [] {
        kael::manual_seed(5);
        return kael::nn::Sequential{std::make_shared<kael::nn::Linear>(4, 8, true, DType::F64),
                                    std::make_shared<kael::nn::ReLU>(),
                                    std::make_shared<kael::nn::Linear>(8, 3, true, DType::F64)};
    };
    kael::nn::Sequential cpu_model = build();
    kael::nn::Sequential device_model = build();
    device_model.to(Device::CUDA);
    kael::optim::AdamW cpu_optimizer(cpu_model.parameters());
    kael::optim::AdamW device_optimizer(device_model.parameters());

    kael::manual_seed(6);
    const Tensor inputs = kael::randn({16, 4}, DType::F64);
    const Tensor targets = kael::randint(0, 3, {16});
    for (int step = 0; step < 20; ++step) {
        cpu_optimizer.zero_grad();
        const Tensor cpu_loss = kael::nn::cross_entropy(cpu_model(inputs), targets);
        cpu_loss.backward();
        kael::optim::clip_grad_norm_(cpu_model.parameters(), 1.0);
        cpu_optimizer.step();

        device_optimizer.zero_grad();
        const Tensor device_loss =
            kael::nn::cross_entropy(device_model(inputs.cuda()), targets.cuda());
        device_loss.backward();
        kael::optim::clip_grad_norm_(device_model.parameters(), 1.0);
        device_optimizer.step();
        expect_matches_cpu(device_loss, cpu_loss, "loss");
    }
    const auto cpu_parameters = cpu_model.parameters();
    const auto device_parameters = device_model.parameters();
    for (std::size_t i = 0; i < cpu_parameters.size(); ++i) {
        expect_matches_cpu(device_parameters[i].detach(), cpu_parameters[i].detach(), "parameter");
    }
}

TEST_F(Cuda, DeviceBlocksAreCached) {
    if (!kael::caching_allocator_enabled()) {
        GTEST_SKIP() << "built without the caching allocator";
    }
    kael::empty_cache();
    {
        const Tensor warm = kael::zeros({1000}).cuda();
    }
    const kael::AllocatorStats before = kael::allocator_stats(Device::CUDA);
    EXPECT_GT(before.cached_bytes, 0);
    {
        const Tensor again = kael::zeros({1000}).cuda();
    }
    const kael::AllocatorStats after = kael::allocator_stats(Device::CUDA);
    EXPECT_EQ(after.hits, before.hits + 1);
    EXPECT_EQ(after.misses, before.misses);
    kael::empty_cache();
    EXPECT_EQ(kael::allocator_stats(Device::CUDA).cached_bytes, 0);
}

TEST_F(Cuda, SequenceOpsMatchCpu) {
    const Tensor x = kael::randn({5, 6});
    expect_matches_cpu(kael::cumsum(x.cuda(), 1), kael::cumsum(x, 1), "cumsum");
    expect_matches_cpu(kael::sort(x.cuda(), 1).values, kael::sort(x, 1).values, "sort");
    expect_matches_cpu(kael::topk(x.cuda(), 2).indices, kael::topk(x, 2).indices, "topk");
    expect_matches_cpu(kael::tril(x.cuda()), kael::tril(x), "tril");
    expect_matches_cpu(kael::triu(x.cuda(), 1), kael::triu(x, 1), "triu");
    EXPECT_EQ(kael::multinomial(kael::ones({3}).cuda(), 2, true).device(), Device::CUDA);
}

TEST_F(Cuda, GptMatchesCpuIncludingTheCacheAndGeneration) {
    kael::nn::GptConfig config;
    config.vocab_size = 11;
    config.max_positions = 16;
    config.layers = 2;
    config.heads = 2;
    config.dimension = 8;
    config.dtype = DType::F64;
    kael::manual_seed(31);
    kael::nn::GPT model(config);
    const Tensor tokens = kael::randint(0, 11, {2, 6});
    const Tensor expected = model(tokens);
    const Tensor generated_on_cpu =
        kael::nn::generate(model, tokens, {.max_new_tokens = 4, .temperature = 0.0});
    model.to(Device::CUDA);
    expect_matches_cpu(model(tokens.cuda()), expected, "logits");
    expect_matches_cpu(
        kael::nn::generate(model, tokens.cuda(), {.max_new_tokens = 4, .temperature = 0.0}),
        generated_on_cpu, "generate");
    expect_matches_cpu(
        kael::nn::generate(model, tokens.cuda(),
                           {.max_new_tokens = 4, .temperature = 0.0, .use_cache = false}),
        generated_on_cpu, "generate without cache");
    kael::sum(model(tokens.cuda())).backward();
    EXPECT_EQ(model.parameters()[0].grad()->device(), Device::CUDA);
}

TEST_F(Cuda, SteppedIndicesWorkOnTheDevice) {
    const Tensor x = kael::randn({4, 2}, DType::F64);
    const Tensor device_x = x.clone().cuda().requires_grad_();
    const Tensor every_other = kael::from_vector<std::int64_t>({3, 0, 1, 0}).slice(0, 0, 4, 2);
    const Tensor selected = kael::index_select(device_x, 0, every_other.cuda());
    expect_matches_cpu(selected.detach(), kael::index_select(x, 0, every_other), "forward");
    kael::sum(selected).backward();
    EXPECT_EQ(device_x.grad()->device(), Device::CUDA);
}
