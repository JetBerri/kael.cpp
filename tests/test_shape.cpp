#include <gtest/gtest.h>
#include "kael/shape.h"

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

TEST(ShapeString, BigDimension){

    kael::Shape test_shape = {500000000000};

    EXPECT_EQ(kael::to_string(test_shape), "[500000000000]");

}