#include <gtest/gtest.h>

#include "kael/autograd/functions.h"
#include "kael/error.h"
#include "kael/kael.h"

using kael::autograd::sum_to_shape;

TEST(SumToShape, LeadingAndStretchedDimensions) {
    const kael::Tensor gradient = kael::ones({4, 2, 3});
    const kael::Tensor reduced = sum_to_shape(gradient, {2, 1});
    EXPECT_EQ(reduced.shape(), (kael::Shape{2, 1}));
    EXPECT_EQ(reduced.at<float>({0, 0}), 12.0f);
}

TEST(SumToShape, SameShapeIsUntouched) {
    const kael::Tensor gradient = kael::ones({2, 3});
    EXPECT_TRUE(sum_to_shape(gradient, {2, 3}).is_same(gradient));
}

TEST(SumToShape, ScalarTargetSumsEverything) {
    const kael::Tensor reduced = sum_to_shape(kael::ones({2, 3}), {});
    EXPECT_EQ(reduced.dim(), 0);
    EXPECT_EQ(reduced.item(), 6.0);
}

TEST(SumToShape, TooFewDimensionsThrows) {
    EXPECT_THROW((void)sum_to_shape(kael::ones({2}), {2, 1}), kael::Error);
}
