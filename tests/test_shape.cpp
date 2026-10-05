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

TEST(ValidateShape, ErrorMessageContainsShape) {
    try {
        kael::validate_shape({2, -2});
        FAIL() << "expected kael::Error";
    } catch (const kael::Error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("[2, -2]"), std::string::npos);
        EXPECT_NE(msg.find("position 1"), std::string::npos);
    }
}