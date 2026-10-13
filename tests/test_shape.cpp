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
TEST(OffsetFromIndex, ContiguousStrides) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index_1 = {0, 0};
    kael::Shape index_2 = {1, 2};
    kael::Shape index_3 = {1, 0};
    EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index_1), 0);
    EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index_2), 5);
    EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index_3), 3);
}

TEST(OffsetFromIndex, BaseOffset) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index = {1, 2};
    EXPECT_EQ(kael::offset_from_index(shape, strides, 10, index), 15);
}

TEST(OffsetFromIndex, NonContiguousStrides) {
    kael::Shape shape = {3, 2};
    kael::Strides strides = {1, 3};
    kael::Shape index = {2, 1};
    EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index), 5); // 2*1 + 1*3
}

TEST(OffsetFromIndex, Scalar) {
    kael::Shape shape = {};
    kael::Strides strides = {};
    kael::Shape index = {};
    EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index), 0);
    EXPECT_EQ(kael::offset_from_index(shape, strides, 7, index), 7);
}

TEST(OffsetFromIndex, IndexOutOfRangeThrows) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index_1 = {2, 0};
    kael::Shape index_2 = {0, 3};
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_1), kael::Error);
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_2), kael::Error);
}

TEST(OffsetFromIndex, NegativeIndexThrows) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index_1 = {-1, 0};
    kael::Shape index_2 = {0, -1};
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_1), kael::Error);
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_2), kael::Error);
}

TEST(OffsetFromIndex, WrongIndexDimsThrows) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index_1 = {1};
    kael::Shape index_2 = {1, 1, 1};
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_1), kael::Error);
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index_2), kael::Error);
}

TEST(OffsetFromIndex, ShapeStridesSizeMismatchThrows) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = {1};
    kael::Shape index = {0, 0};
    EXPECT_THROW(kael::offset_from_index(shape, strides, 0, index), kael::Error);
}

TEST(OffsetFromIndex, ErrorMessageContainsValues) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = kael::contiguous_strides(shape);
    kael::Shape index = {1, 5};
    try {
        kael::offset_from_index(shape, strides, 0, index);
        FAIL() << "expected kael::Error";
    } catch (const kael::Error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("[2, 3]"), std::string::npos);
        EXPECT_NE(msg.find("index 5"), std::string::npos);
        EXPECT_NE(msg.find("dimension 1"), std::string::npos);
    }
}

TEST(IndexFromLinear, RowMajorOrder) {
    EXPECT_EQ(kael::index_from_linear(0, {2, 3}), (kael::Shape{0, 0}));
    EXPECT_EQ(kael::index_from_linear(4, {2, 3}), (kael::Shape{1, 1}));
    EXPECT_EQ(kael::index_from_linear(5, {2, 3}), (kael::Shape{1, 2}));
    EXPECT_EQ(kael::index_from_linear(13, {2, 3, 4}), (kael::Shape{1, 0, 1})); // 1*12 + 0*4 + 1
    EXPECT_EQ(kael::index_from_linear(5, {3, 1, 2}), (kael::Shape{2, 0, 1}));  // dimension of 1
}

TEST(IndexFromLinear, Scalar) {
    // a scalar has one element, so only linear 0 is valid and the index is empty
    EXPECT_EQ(kael::index_from_linear(0, {}), kael::Shape{});
    EXPECT_THROW(kael::index_from_linear(1, {}), kael::Error);
}

TEST(IndexFromLinear, RoundTripWithOffsetFromIndex) {
    kael::Shape shape = {2, 3, 4};
    kael::Strides strides = kael::contiguous_strides(shape);
    for (std::int64_t linear = 0; linear < kael::numel(shape); linear++) {
        kael::Shape index = kael::index_from_linear(linear, shape);
        EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index), linear);
    }
}

TEST(IndexFromLinear, OutOfRangeThrows) {
    EXPECT_THROW(kael::index_from_linear(-1, {2, 3}), kael::Error);
    EXPECT_THROW(kael::index_from_linear(6, {2, 3}), kael::Error);
    // numel is 0, so no linear index is valid (and no modulo by zero is reached)
    EXPECT_THROW(kael::index_from_linear(0, {2, 0, 3}), kael::Error);
    // negative shape is rejected by numel
    EXPECT_THROW(kael::index_from_linear(0, {-1, 3}), kael::Error);
}

TEST(BroadcastShapes, ValidCases) {
    EXPECT_EQ(kael::broadcast_shapes({3}, {2, 3}), (kael::Shape{2, 3}));
    EXPECT_EQ(kael::broadcast_shapes({2, 1}, {1, 3}), (kael::Shape{2, 3}));
    EXPECT_EQ(kael::broadcast_shapes({5, 1, 4}, {3, 1}), (kael::Shape{5, 3, 4}));
    EXPECT_EQ(kael::broadcast_shapes({}, {4}), (kael::Shape{4}));
    EXPECT_EQ(kael::broadcast_shapes({}, {}), kael::Shape{});
    EXPECT_EQ(kael::broadcast_shapes({0}, {1}), (kael::Shape{0})); // 0 with 1 gives 0, like NumPy
}

TEST(BroadcastShapes, Symmetry) {
    const std::vector<std::pair<kael::Shape, kael::Shape>> pairs = {
        {{3}, {2, 3}},
        {{2, 1}, {1, 3}},
        {{5, 1, 4}, {3, 1}},
        {{}, {4}},
        {{}, {}},
        {{0}, {1}},
    };
    for (const auto& [a, b] : pairs) {
        EXPECT_EQ(kael::broadcast_shapes(a, b), kael::broadcast_shapes(b, a));
    }
}

TEST(BroadcastShapes, IncompatibleThrows) {
    EXPECT_THROW(kael::broadcast_shapes({2, 3}, {3, 2}), kael::Error);
    EXPECT_THROW(kael::broadcast_shapes({0}, {3}), kael::Error); // 0 only broadcasts with 0 or 1
}

TEST(BroadcastShapes, NegativeDimensionThrows) {
    EXPECT_THROW(kael::broadcast_shapes({-1, 3}, {3}), kael::Error);
    EXPECT_THROW(kael::broadcast_shapes({3}, {2, -1}), kael::Error);
}

TEST(BroadcastShapes, ErrorMessageContainsShapes) {
    try {
        kael::broadcast_shapes({2, 3}, {3, 2});
        FAIL() << "expected kael::Error";
    } catch (const kael::Error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("[2, 3]"), std::string::npos);
        EXPECT_NE(msg.find("[3, 2]"), std::string::npos);
    }
}

TEST(BroadcastStrides, ValidCases) {
    EXPECT_EQ(kael::broadcast_strides({3}, {1}, {2, 3}),
              (kael::Strides{0, 1}));

    EXPECT_EQ(kael::broadcast_strides({2, 1}, {1, 1}, {2, 4}),
              (kael::Strides{1, 0}));

    EXPECT_EQ(kael::broadcast_strides({}, {}, {2, 2}),
              (kael::Strides{0, 0}));
}

TEST(BroadcastStrides, SameShape) {
    kael::Shape shape = {2, 3};
    kael::Strides strides = {3, 1};

    EXPECT_EQ(kael::broadcast_strides(shape, strides, shape), strides);
}

TEST(BroadcastStrides, MoreDimensionsThanTargetThrows) {
    EXPECT_THROW(kael::broadcast_strides({2, 3}, {3, 1}, {3}),
                 kael::Error);
}

TEST(BroadcastStrides, IncompatibleDimensionsThrows) {
    EXPECT_THROW(kael::broadcast_strides({2}, {1}, {3}),
                 kael::Error);
}

TEST(BroadcastStrides, ShapeStridesSizeMismatchThrows) {
    EXPECT_THROW(kael::broadcast_strides({2, 3}, {1}, {2, 3}),
                 kael::Error);
}

TEST(BroadcastStrides, ErrorMessageContainsShapes) {
    try {
        kael::broadcast_strides({2}, {1}, {3});
        FAIL() << "expected kael::Error";
    } catch (const kael::Error& e) {
        std::string msg = e.what();

        EXPECT_NE(msg.find("[2]"), std::string::npos);
        EXPECT_NE(msg.find("[3]"), std::string::npos);
    }
}

TEST(BroadcastStrides, SemanticBroadcastRows) {
    kael::Shape shape = {3};
    kael::Strides strides = {1};
    kael::Shape target = {2, 3};

    std::vector<float> data = {0.0f, 1.0f, 2.0f};

    kael::Strides broadcasted = kael::broadcast_strides(shape, strides, target);

    for (std::int64_t linear = 0; linear < kael::numel(target); linear++) {
        kael::Shape index = kael::index_from_linear(linear, target);

        std::int64_t offset = kael::offset_from_index(target, broadcasted, 0, index);

        EXPECT_FLOAT_EQ(data.at(offset), data.at(index[1]));
    }
}

TEST(BroadcastStrides, SemanticBroadcastColumns) {
    kael::Shape shape = {2, 1};
    kael::Strides strides = {1, 1};
    kael::Shape target = {2, 4};

    std::vector<float> data = {0.0f, 1.0f};

    kael::Strides broadcasted = kael::broadcast_strides(shape, strides, target);

    for (std::int64_t linear = 0; linear < kael::numel(target); linear++) {
        kael::Shape index = kael::index_from_linear(linear, target);

        std::int64_t offset = kael::offset_from_index(target, broadcasted, 0, index);

        EXPECT_FLOAT_EQ(data.at(offset), data.at(index[0]));
    }
}