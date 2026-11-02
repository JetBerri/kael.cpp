#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/kernels.h"

#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::Tensor;
using kael::test::values;

namespace {

constexpr auto kAdd = [](auto x, auto y) { return x + y; };

Tensor add_reference(const Tensor& a, const Tensor& b) {
    const Shape shape = kael::broadcast_shapes(a.shape(), b.shape());
    const Tensor expanded_a = a.expand(shape);
    const Tensor expanded_b = b.expand(shape);
    Tensor out = kael::empty(shape, DType::F64);
    for (std::int64_t linear = 0; linear < out.numel(); ++linear) {
        const Shape index = kael::index_from_linear(linear, shape);
        out.at<double>(index) = expanded_a.at<double>(index) + expanded_b.at<double>(index);
    }
    return out;
}

class RandomCases {
public:
    explicit RandomCases(std::uint64_t seed) : generator_(seed) {}

    std::int64_t uniform(std::int64_t low, std::int64_t high) {
        return std::uniform_int_distribution<std::int64_t>(low, high)(generator_);
    }

    Shape shape() {
        Shape result(static_cast<std::size_t>(uniform(0, 4)));
        for (std::int64_t& dim : result) {
            dim = uniform(0, 5);
        }
        return result;
    }

    Shape broadcastable_from(const Shape& shape) {
        Shape result;
        for (const std::int64_t dim : shape) {
            result.push_back(uniform(0, 2) == 0 ? 1 : dim);
        }
        result.erase(result.begin(),
                     result.begin() + uniform(0, static_cast<std::int64_t>(result.size())));
        return result;
    }

    Tensor tensor(const Shape& shape) {
        const std::int64_t style = uniform(0, 3);
        if (style == 0 && shape.size() >= 2) {
            return transposed(shape);
        }
        if (style == 1 && !shape.empty()) {
            return stepped(shape);
        }
        if (style == 2 && !shape.empty() && shape[0] == 1) {
            return expanded(shape);
        }
        return iota(shape);
    }

private:
    Tensor iota(const Shape& shape) {
        const auto count = static_cast<double>(kael::numel(shape));
        return kael::arange(0, count, 1, DType::F64).reshape(shape);
    }

    Tensor transposed(const Shape& shape) {
        Shape swapped = shape;
        std::swap(swapped.front(), swapped.back());
        return iota(swapped).transpose(0, -1);
    }

    Tensor stepped(const Shape& shape) {
        Shape doubled = shape;
        doubled.back() *= 2;
        return iota(doubled).slice(-1, 0, doubled.back(), 2);
    }

    Tensor expanded(const Shape& shape) {
        Shape single = shape;
        single.front() = 1;
        return iota(single).expand(shape);
    }

    std::mt19937_64 generator_;
};

} // namespace

TEST(Kernels, RandomCasesMatchTheNaiveReference) {
    RandomCases random(2024);
    for (int i = 0; i < 500; ++i) {
        const Shape shape_a = random.shape();
        const Shape shape_b = random.broadcastable_from(shape_a);
        const Tensor a = random.tensor(shape_a);
        const Tensor b = random.tensor(shape_b);
        SCOPED_TRACE("case " + std::to_string(i) + " a=" + kael::to_string(shape_a) +
                     " b=" + kael::to_string(shape_b));

        const Tensor result = kael::binary_kernel(a, b, kAdd);
        const Tensor reference = add_reference(a, b);
        EXPECT_EQ(result.shape(), reference.shape());
        EXPECT_EQ(values<double>(result), values<double>(reference));
    }
}

TEST(Kernels, UnaryAppliesToEveryElement) {
    auto t = kael::arange(0, 4).reshape({2, 2}).T();
    auto out = kael::unary_kernel(t, [](auto x) { return x * 2; });
    EXPECT_EQ(out.dtype(), DType::F32);
    EXPECT_EQ(values<float>(out), (std::vector<float>{0, 4, 2, 6}));
}

TEST(Kernels, BinaryPromotesInputs) {
    auto a = kael::arange(0, 3, 1, DType::I64);
    auto b = kael::full({3}, 0.5, DType::F32);
    auto out = kael::binary_kernel(a, b, kAdd);
    EXPECT_EQ(out.dtype(), DType::F32);
    EXPECT_EQ(values<float>(out), (std::vector<float>{0.5f, 1.5f, 2.5f}));
}

TEST(Kernels, TernaryBroadcastsAllThree) {
    auto a = kael::arange(0, 2).reshape({2, 1});
    auto b = kael::arange(0, 3);
    auto c = kael::scalar(10);
    auto out = kael::ternary_kernel(a, b, c, [](auto x, auto y, auto z) { return x + y + z; });
    EXPECT_EQ(out.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(out), (std::vector<float>{10, 11, 12, 11, 12, 13}));
}

TEST(Kernels, ComparisonOutputsBool) {
    auto a = kael::arange(0, 3);
    auto b = kael::full({3}, 1.0, DType::I64);
    auto out = kael::comparison_kernel(a, b, [](auto x, auto y) { return x < y; });
    EXPECT_EQ(out.dtype(), DType::Bool);
    EXPECT_EQ(values<std::uint8_t>(out), (std::vector<std::uint8_t>{1, 0, 0}));
}

TEST(Kernels, GivenOutputWithWrongShapeThrows) {
    auto out = kael::zeros({2});
    EXPECT_THROW(kael::binary_kernel_out(out, kael::zeros({3}), kael::zeros({3}), kAdd),
                 kael::Error);
}

TEST(Kernels, OutputCanBeOneOfTheInputs) {
    auto a = kael::arange(0, 3);
    kael::binary_kernel_out(a, a, kael::full({3}, 1.0), kAdd);
    EXPECT_EQ(values<float>(a), (std::vector<float>{1, 2, 3}));
}

TEST(Kernels, UnaryOutWritesThroughViews) {
    auto base = kael::zeros({2, 2});
    auto column = base.select(1, 1);
    kael::unary_kernel_out(column, kael::full({2}, 7.0), [](auto x) { return x; });
    EXPECT_EQ(values<float>(base), (std::vector<float>{0, 7, 0, 7}));
}

TEST(Kernels, EmptyAndScalarInputs) {
    EXPECT_EQ(kael::binary_kernel(kael::zeros({0, 2}), kael::zeros({2}), kAdd).shape(),
              (Shape{0, 2}));
    EXPECT_EQ(kael::binary_kernel(kael::scalar(1), kael::scalar(2), kAdd).item<float>(), 3.0f);
}

TEST(StridedCopy, ConvertsThroughIterator) {
    auto t = kael::arange(0, 6, 1, DType::I64).reshape({2, 3}).T();
    EXPECT_EQ(values<double>(t.to(DType::F64)), (std::vector<double>{0, 3, 1, 4, 2, 5}));
    EXPECT_EQ(values<std::int64_t>(t.clone()), (std::vector<std::int64_t>{0, 3, 1, 4, 2, 5}));
}
