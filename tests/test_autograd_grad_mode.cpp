#include <stdexcept>

#include <gtest/gtest.h>

#include "kael/autograd/grad_mode.h"

using kael::autograd::EnableGradGuard;
using kael::autograd::GradMode;
using kael::autograd::NoGradGuard;

TEST(GradMode, EnabledByDefault) {
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(GradMode, NoGradGuardDisablesAndRestores) {
    {
        NoGradGuard guard;
        EXPECT_FALSE(GradMode::is_enabled());
    }
    EXPECT_TRUE(GradMode::is_enabled());
}

TEST(GradMode, NestedGuardsRestoreInOrder) {
    NoGradGuard outer;
    {
        EnableGradGuard enabled;
        EXPECT_TRUE(GradMode::is_enabled());
        {
            NoGradGuard inner;
            EXPECT_FALSE(GradMode::is_enabled());
        }
        EXPECT_TRUE(GradMode::is_enabled());
    }
    EXPECT_FALSE(GradMode::is_enabled());
}

TEST(GradMode, ExceptionRestoresState) {
    try {
        NoGradGuard guard;
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }
    EXPECT_TRUE(GradMode::is_enabled());
}
