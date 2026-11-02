#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "support/allclose.h"
#include "support/npy.h"
#include "support/values.h"

using kael::DType;
using kael::Shape;
using kael::test::allclose;
using kael::test::read_npy;
using kael::test::values;

namespace {

const std::string kFixtures = std::string(KAEL_GOLDEN_DIR) + "/npy/";
const double kNan = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

} // namespace

TEST(Allclose, ToleranceDependsOnDtype) {
    EXPECT_TRUE(allclose(kael::from_vector({1.0f}), kael::from_vector({1.000001f})));
    EXPECT_FALSE(allclose(kael::from_vector({1.0f}), kael::from_vector({1.01f})));
    EXPECT_TRUE(allclose(kael::from_vector({1.0}), kael::from_vector({1.0 + 1e-13})));
    EXPECT_FALSE(allclose(kael::from_vector({1.0}), kael::from_vector({1.0 + 1e-9})));
}

TEST(Allclose, AbsoluteToleranceCoversValuesNearZero) {
    EXPECT_TRUE(allclose(kael::from_vector({0.0f}), kael::from_vector({1e-7f})));
    EXPECT_FALSE(allclose(kael::from_vector({0.0f}), kael::from_vector({1e-3f})));
}

TEST(Allclose, IntegersAreExact) {
    auto a = kael::from_vector(std::vector<std::int64_t>{1});
    auto b = kael::from_vector(std::vector<std::int64_t>{2});
    EXPECT_TRUE(allclose(a, a));
    EXPECT_FALSE(allclose(a, b));
}

TEST(Allclose, NanNeedsOption) {
    auto a = kael::from_vector({kNan});
    EXPECT_FALSE(allclose(a, a));
    EXPECT_TRUE(allclose(a, a, {.equal_nan = true}));
    EXPECT_FALSE(allclose(a, kael::from_vector({1.0}), {.equal_nan = true}));
}

TEST(Allclose, InfinitiesMustMatch) {
    auto plus = kael::from_vector({kInf});
    auto minus = kael::from_vector({-kInf});
    EXPECT_TRUE(allclose(plus, plus));
    EXPECT_FALSE(allclose(plus, minus));
    EXPECT_FALSE(allclose(plus, kael::from_vector({1e300})));
}

TEST(Allclose, ShapeAndDtypeMustMatch) {
    EXPECT_FALSE(allclose(kael::zeros({2}), kael::zeros({3})));
    EXPECT_FALSE(allclose(kael::zeros({2}), kael::zeros({2}, DType::F64)));
}

TEST(Allclose, FailureNamesTheElement) {
    auto a = kael::from_vector({1.0f, 2.0f});
    auto b = kael::from_vector({1.0f, 3.0f});
    const std::string message = allclose(a, b).message();
    EXPECT_NE(message.find("element [1]"), std::string::npos);
    EXPECT_NE(message.find("2"), std::string::npos);
    EXPECT_NE(message.find("3"), std::string::npos);
}

TEST(Npy, ReadsEveryDtype) {
    auto f32 = read_npy(kFixtures + "f32.npy");
    EXPECT_EQ(f32.shape(), (Shape{2, 3}));
    EXPECT_EQ(values<float>(f32), (std::vector<float>{0, 1.5f, 3, 4.5f, 6, 7.5f}));

    auto f64 = read_npy(kFixtures + "f64.npy");
    EXPECT_EQ(f64.shape(), (Shape{3}));
    EXPECT_EQ(values<double>(f64), (std::vector<double>{0.25, 0.5, 0.75}));

    auto i64 = read_npy(kFixtures + "i64.npy");
    EXPECT_EQ(values<std::int64_t>(i64), (std::vector<std::int64_t>{-1, 2}));

    auto b1 = read_npy(kFixtures + "bool.npy");
    EXPECT_EQ(b1.dtype(), DType::Bool);
    EXPECT_EQ(values<std::uint8_t>(b1), (std::vector<std::uint8_t>{1, 0, 1}));
}

TEST(Npy, ReadsScalarAndEmpty) {
    auto scalar = read_npy(kFixtures + "scalar.npy");
    EXPECT_EQ(scalar.dim(), 0);
    EXPECT_EQ(scalar.item<float>(), 2.5f);
    EXPECT_EQ(read_npy(kFixtures + "empty.npy").shape(), (Shape{0, 3}));
}

TEST(Npy, RejectsWhatItCannotRead) {
    EXPECT_THROW(read_npy(kFixtures + "missing.npy"), kael::Error);
    EXPECT_THROW(read_npy(kFixtures + "fortran.npy"), kael::Error);
    EXPECT_THROW(read_npy(kFixtures + "f16.npy"), kael::Error);
    EXPECT_THROW(read_npy(kFixtures + "not_npy.npy"), kael::Error);
}
