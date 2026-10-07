#include <gtest/gtest.h>
#include "kael/shape.h"
#include "kael/error.h"

TEST(ShapeString, NormalCase){
    kael::Shape test_shape = {1, 2, 3, 4};
    EXPECT_EQ(kael::to_string(test_shape), "[1, 2, 3, 4]");
}

TEST(ShapeString, EmptyCase){
    kael::Shape test_shape = {};
    EXPECT_EQ(kael::to_string(test_shape), "[]");
}

TEST(ShapeString, OneDimension){
    kael::Shape test_shape = {1};
    EXPECT_EQ(kael::to_string(test_shape), "[1]");
}

TEST(ShapeString, ZeroDimension){
    kael::Shape test_shape = {3, 0, 5};
    EXPECT_EQ(kael::to_string(test_shape), "[3, 0, 5]");
}

TEST(ShapeString, BigDimension){
    kael::Shape test_shape = {500000000000};
    EXPECT_EQ(kael::to_string(test_shape), "[500000000000]");
}

TEST(ValidateShape, NoError){
    kael::Shape test_shape_1 = {};
    kael::Shape test_shape_2 = {2, 3};
    kael::Shape test_shape_3 = {0};
    kael::Shape test_shape_4 = {3, 0, 2};
    EXPECT_NO_THROW(kael::validate_shape(test_shape_1));
    EXPECT_NO_THROW(kael::validate_shape(test_shape_2));
    EXPECT_NO_THROW(kael::validate_shape(test_shape_3));
    EXPECT_NO_THROW(kael::validate_shape(test_shape_4));
}

TEST(ValidateShape, ThrowError){
    kael::Shape test_shape_1 = {-5};
    kael::Shape test_shape_2 = {-1, 3};
    kael::Shape test_shape_3 = {2, -2};
    kael::Shape test_shape_4 = {0, -1};
    EXPECT_THROW(kael::validate_shape(test_shape_1), kael::Error);
    EXPECT_THROW(kael::validate_shape(test_shape_2), kael::Error);
    EXPECT_THROW(kael::validate_shape(test_shape_3), kael::Error);
    EXPECT_THROW(kael::validate_shape(test_shape_4), kael::Error);
}

TEST(ValidateShape, ErrorMessageContainsShape){
    try {
        kael::validate_shape({2, -2});
        FAIL() << "expected kael::Error";
    } catch (const kael::Error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("[2, -2]"), std::string::npos);
        EXPECT_NE(msg.find("position 1"), std::string::npos);
    }
}

TEST(Numel, RegularCase){
    kael::Shape test_shape_1 = {};
    kael::Shape test_shape_2 = {5};
    kael::Shape test_shape_3 = {2, 3, 4};
    kael::Shape test_shape_4 = {3, 0, 2};
    EXPECT_EQ(kael::numel(test_shape_1), 1);
    EXPECT_EQ(kael::numel(test_shape_2), 5);
    EXPECT_EQ(kael::numel(test_shape_3), 24);
    EXPECT_EQ(kael::numel(test_shape_4), 0);
}

TEST(Numel, NegativeShape){
    kael::Shape test_shape_1 = {-1};
    kael::Shape test_shape_2 = {1, -2};
    EXPECT_THROW(kael::numel(test_shape_1), kael::Error);
    EXPECT_THROW(kael::numel(test_shape_2), kael::Error);
}

TEST(Numel, Overflow){
    // 2^40 * 2^40 = 2^80, does not fit in int64 (max ~ 2^63)
    EXPECT_THROW(kael::numel({1LL << 40, 1LL << 40}), kael::Error);

    // Boundary: the largest int64 alone fits, but doubling it does not
    const std::int64_t max = std::numeric_limits<std::int64_t>::max();
    EXPECT_EQ(kael::numel({max}), max);
    EXPECT_THROW(kael::numel({max, 2}), kael::Error);

    // 2^31 * 2^31 = 2^62 still fits
    EXPECT_EQ(kael::numel({1LL << 31, 1LL << 31}), 1LL << 62);
}

TEST(ContiguousStrides, RegularShapes) {
    EXPECT_EQ(kael::contiguous_strides({2, 3, 4}), (kael::Strides{12, 4, 1}));
    EXPECT_EQ(kael::contiguous_strides({5}), (kael::Strides{1}));
}

TEST(ContiguousStrides, Scalar) {
    EXPECT_EQ(kael::contiguous_strides({}), kael::Strides{});
}

TEST(ContiguousStrides, DimensionOfOne) {
    EXPECT_EQ(kael::contiguous_strides({3, 1, 2}), (kael::Strides{2, 2, 1}));
}

TEST(ContiguousStrides, DimensionOfZero) {
    EXPECT_EQ(kael::contiguous_strides({2, 0, 3}), (kael::Strides{3, 3, 1}));
}

TEST(ContiguousStrides, NegativeShapeThrows) {
    EXPECT_THROW(kael::contiguous_strides({-1}), kael::Error);
}