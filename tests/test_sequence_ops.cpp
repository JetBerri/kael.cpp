#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "kael/autograd/gradcheck.h"
#include "kael/kael.h"

#include "support/allclose.h"

using kael::DType;
using kael::Tensor;
using Inputs = std::vector<Tensor>;

namespace {

Tensor doubles(const std::vector<double>& values, const kael::Shape& shape) {
    return kael::from_vector(values, shape);
}

Tensor integers(const std::vector<std::int64_t>& values, const kael::Shape& shape) {
    return kael::from_vector(values, shape);
}

Tensor differentiable(const kael::Shape& shape) {
    Tensor x = kael::randn(shape, DType::F64);
    x.requires_grad_();
    return x;
}

} // namespace

TEST(Cumsum, AlongEachDimension) {
    const Tensor x = doubles({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_TRUE(kael::equal(kael::cumsum(x, 1), doubles({1, 3, 6, 4, 9, 15}, {2, 3})));
    EXPECT_TRUE(kael::equal(kael::cumsum(x, 0), doubles({1, 2, 3, 5, 7, 9}, {2, 3})));
    EXPECT_TRUE(kael::equal(kael::cumsum(x, -1), kael::cumsum(x, 1)));
    EXPECT_TRUE(kael::cumsum(x.T(), 1).is_contiguous());
}

TEST(Cumsum, PromotesIntegersAndBools) {
    EXPECT_EQ(kael::cumsum(kael::ones({3}, DType::Bool), 0).dtype(), DType::I64);
    EXPECT_EQ(kael::cumsum(kael::ones({3}, DType::Bool), 0).at<std::int64_t>({2}), 3);
    EXPECT_EQ(kael::cumsum(kael::ones({3}, DType::I64), 0).dtype(), DType::I64);
}

TEST(Cumsum, EmptyAndBadDimension) {
    EXPECT_EQ(kael::cumsum(kael::zeros({0, 2}), 0).shape(), (kael::Shape{0, 2}));
    EXPECT_THROW((void)kael::cumsum(kael::zeros({2}), 3), kael::IndexError);
}

TEST(Cumsum, Gradcheck) {
    kael::manual_seed(1);
    for (const std::int64_t dim : {0, 1}) {
        const auto result = kael::autograd::gradcheck(
            [&](const Inputs& in) { return kael::cumsum(in[0], dim); }, {differentiable({3, 4})});
        EXPECT_TRUE(result.passed) << result.report;
    }
}

TEST(Sort, ValuesAndIndices) {
    const Tensor x = doubles({3, 1, 2, 9, 7, 8}, {2, 3});
    const auto ascending = kael::sort(x);
    EXPECT_TRUE(kael::equal(ascending.values, doubles({1, 2, 3, 7, 8, 9}, {2, 3})));
    EXPECT_TRUE(kael::equal(ascending.indices, integers({1, 2, 0, 1, 2, 0}, {2, 3})));
    const auto descending = kael::sort(x, 0, true);
    EXPECT_TRUE(kael::equal(descending.values, doubles({9, 7, 8, 3, 1, 2}, {2, 3})));
    EXPECT_TRUE(kael::equal(descending.indices, integers({1, 1, 1, 0, 0, 0}, {2, 3})));
}

TEST(Sort, PlacesNanLastWhenAscendingAndFirstWhenDescending) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Tensor x = doubles({nan, 1, 2}, {3});
    EXPECT_EQ(kael::sort(x).indices.at<std::int64_t>({2}), 0);
    EXPECT_EQ(kael::sort(x, -1, true).indices.at<std::int64_t>({0}), 0);
}

TEST(Sort, EqualValuesKeepTheirOrder) {
    const Tensor x = doubles({1, 1, 0, 1}, {4});
    EXPECT_TRUE(kael::equal(kael::sort(x).indices, integers({2, 0, 1, 3}, {4})));
}

TEST(Sort, WorksOnIntegersAndBools) {
    EXPECT_TRUE(
        kael::equal(kael::sort(integers({3, -1, 2}, {3})).values, integers({-1, 2, 3}, {3})));
    const Tensor flags = kael::from_vector(std::vector<bool>{true, false}, {2});
    EXPECT_EQ(kael::sort(flags).indices.at<std::int64_t>({0}), 1);
}

TEST(Sort, GradientFollowsThePermutation) {
    Tensor x = doubles({3, 1, 2}, {3});
    x.requires_grad_();
    const auto sorted = kael::sort(x);
    kael::sum(sorted.values * doubles({1, 10, 100}, {3})).backward();
    EXPECT_TRUE(kael::equal(*x.grad(), doubles({100, 1, 10}, {3})));
}

TEST(Sort, Gradcheck) {
    kael::manual_seed(2);
    const auto result = kael::autograd::gradcheck(
        [](const Inputs& in) { return kael::sort(in[0], 1).values; }, {differentiable({2, 4})});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(Topk, LargestAndSmallest) {
    const Tensor x = doubles({1, 5, 3, 4, 2, 6}, {2, 3});
    const auto top = kael::topk(x, 2);
    EXPECT_TRUE(kael::equal(top.values, doubles({5, 3, 6, 4}, {2, 2})));
    EXPECT_TRUE(kael::equal(top.indices, integers({1, 2, 2, 0}, {2, 2})));
    const auto bottom = kael::topk(x, 1, -1, false);
    EXPECT_TRUE(kael::equal(bottom.values, doubles({1, 2}, {2, 1})));
}

TEST(Topk, OtherDimensionAndLimits) {
    const Tensor x = doubles({1, 5, 3, 4, 2, 6}, {2, 3});
    EXPECT_TRUE(kael::equal(kael::topk(x, 1, 0).values, doubles({4, 5, 6}, {1, 3})));
    EXPECT_EQ(kael::topk(x, 0).values.shape(), (kael::Shape{2, 0}));
    EXPECT_THROW((void)kael::topk(x, 4), kael::ValueError);
    EXPECT_THROW((void)kael::topk(x, -1), kael::ValueError);
}

TEST(Triangular, UpperAndLowerWithDiagonal) {
    const Tensor x = doubles({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    EXPECT_TRUE(kael::equal(kael::tril(x), doubles({1, 0, 0, 4, 5, 0, 7, 8, 9}, {3, 3})));
    EXPECT_TRUE(kael::equal(kael::triu(x), doubles({1, 2, 3, 0, 5, 6, 0, 0, 9}, {3, 3})));
    EXPECT_TRUE(kael::equal(kael::triu(x, 1), doubles({0, 2, 3, 0, 0, 6, 0, 0, 0}, {3, 3})));
    EXPECT_TRUE(kael::equal(kael::tril(x, -1), doubles({0, 0, 0, 4, 0, 0, 7, 8, 0}, {3, 3})));
}

TEST(Triangular, BatchesRectanglesAndDtypes) {
    const Tensor batch = kael::ones({2, 2, 3}, DType::I64);
    const Tensor upper = kael::triu(batch);
    EXPECT_EQ(upper.dtype(), DType::I64);
    EXPECT_EQ(upper.at<std::int64_t>({1, 1, 0}), 0);
    EXPECT_EQ(upper.at<std::int64_t>({1, 1, 2}), 1);
    EXPECT_EQ(kael::tril(kael::ones({2, 2}, DType::Bool)).dtype(), DType::Bool);
    EXPECT_THROW((void)kael::tril(kael::ones({2})), kael::ValueError);
}

TEST(Triangular, Gradcheck) {
    kael::manual_seed(3);
    const auto result = kael::autograd::gradcheck(
        [](const Inputs& in) { return kael::tril(in[0], -1) + kael::triu(in[0], 1) * 2; },
        {differentiable({3, 3})});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(Multinomial, ShapesAndDtype) {
    kael::manual_seed(4);
    const Tensor probabilities = kael::full({4, 5}, 0.2);
    const Tensor drawn = kael::multinomial(probabilities, 3, true);
    EXPECT_EQ(drawn.shape(), (kael::Shape{4, 3}));
    EXPECT_EQ(drawn.dtype(), DType::I64);
    EXPECT_EQ(kael::multinomial(kael::full({5}, 0.2), 2, true).shape(), kael::Shape{2});
}

TEST(Multinomial, FollowsTheDistribution) {
    kael::manual_seed(5);
    const Tensor drawn = kael::multinomial(doubles({0.1, 0.9, 0.0}, {3}), 20000, true);
    const double mean = kael::mean(drawn.to(DType::F64)).item<double>();
    EXPECT_NEAR(mean, 0.9, 0.02);
    EXPECT_LE(kael::max(drawn).item<std::int64_t>(), 1);
}

TEST(Multinomial, DoesNotNeedNormalizedWeights) {
    kael::manual_seed(6);
    const Tensor drawn = kael::multinomial(doubles({0, 0, 5}, {3}), 10, true);
    EXPECT_EQ(kael::min(drawn).item<std::int64_t>(), 2);
}

TEST(Multinomial, WithoutReplacementNeverRepeats) {
    kael::manual_seed(7);
    const Tensor drawn = kael::multinomial(kael::ones({6}), 6);
    EXPECT_TRUE(kael::equal(kael::sort(drawn).values, kael::arange(0, 6, 1, DType::I64)));
    EXPECT_THROW((void)kael::multinomial(doubles({1, 0}, {2}), 2), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(kael::ones({2}), 3), kael::ValueError);
}

TEST(Multinomial, IsReproducibleWithASeed) {
    kael::manual_seed(8);
    const Tensor first = kael::multinomial(kael::ones({100}), 10, true);
    kael::manual_seed(8);
    EXPECT_TRUE(kael::equal(first, kael::multinomial(kael::ones({100}), 10, true)));
}

TEST(Multinomial, RejectsInvalidInput) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)kael::multinomial(doubles({-1, 2}, {2}), 1, true), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(doubles({nan, 2}, {2}), 1, true), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(doubles({0, 0}, {2}), 1, true), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(kael::ones({2, 2, 2}), 1, true), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(kael::ones({2}), 0, true), kael::ValueError);
    EXPECT_THROW((void)kael::multinomial(kael::ones({2}, DType::I64), 1, true), kael::TypeError);
}

TEST(FastIndexing, IndexSelectMatchesEveryLayout) {
    kael::manual_seed(9);
    const Tensor x = kael::randn({5, 4, 3});
    const Tensor rows = integers({4, 0, 0, 2}, {4});
    for (const std::int64_t dim : {0, 1, 2}) {
        const Tensor picks = integers({1, 0}, {2});
        const Tensor selected = kael::index_select(x, dim, picks);
        for (std::int64_t i = 0; i < 2; ++i) {
            EXPECT_TRUE(kael::equal(selected.narrow(dim, i, 1),
                                    x.narrow(dim, picks.at<std::int64_t>({i}), 1)));
        }
    }
    const Tensor selected = kael::index_select(x, 0, rows);
    EXPECT_TRUE(kael::equal(selected.select(0, 3), x.select(0, 2)));
    const Tensor swapped = x.transpose(0, 1);
    EXPECT_TRUE(kael::equal(kael::index_select(swapped, 0, integers({1}, {1})).select(0, 0),
                            swapped.select(0, 1)));
    EXPECT_THROW((void)kael::index_select(x, 0, integers({5}, {1})), kael::IndexError);
}

TEST(FastIndexing, GatherAndScatterMatchByHand) {
    kael::manual_seed(10);
    const Tensor x = kael::randn({4, 5});
    const Tensor picks = kael::randint(0, 5, {4, 3});
    const Tensor gathered = kael::gather(x, 1, picks);
    Tensor scattered = kael::zeros({4, 5});
    const Tensor source = kael::ones({4, 3});
    scattered.scatter_add_(1, picks, source);
    double total = 0.0;
    for (std::int64_t i = 0; i < 4; ++i) {
        for (std::int64_t j = 0; j < 3; ++j) {
            const std::int64_t column = picks.at<std::int64_t>({i, j});
            EXPECT_EQ(gathered.at<float>({i, j}), x.at<float>({i, column}));
            total += scattered.at<float>({i, column}) > 0 ? 0 : 1;
        }
    }
    EXPECT_EQ(total, 0.0);
    EXPECT_EQ(kael::sum(scattered).item<float>(), 12.0f);
    EXPECT_THROW((void)kael::gather(x, 1, kael::full({1, 1}, 9, DType::I64)), kael::IndexError);
    EXPECT_THROW((void)kael::gather(x, 1, kael::full({1, 1}, -1, DType::I64)), kael::IndexError);
}

TEST(FastIndexing, BackwardAcceptsSteppedIndices) {
    Tensor x = kael::full({4, 2}, 1.0, DType::F64);
    x.requires_grad_();
    const Tensor every_other = integers({3, 0, 1, 0}, {4}).slice(0, 0, 4, 2);
    EXPECT_FALSE(every_other.is_contiguous());
    kael::sum(kael::index_select(x, 0, every_other)).backward();
    EXPECT_EQ(x.grad()->at<double>({3, 0}), 1.0);
    EXPECT_EQ(x.grad()->at<double>({1, 1}), 1.0);
    EXPECT_EQ(x.grad()->at<double>({0, 0}), 0.0);
}
