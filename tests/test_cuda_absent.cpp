#include <gtest/gtest.h>

#include "kael/kael.h"

TEST(CudaAbsent, MovingToADeviceThatIsMissingFailsClearly) {
    if (kael::cuda_is_available()) {
        GTEST_SKIP() << "a CUDA device is present";
    }
    EXPECT_THROW((void)kael::ones({2}).cuda(), kael::Error);
    EXPECT_THROW((void)kael::zeros({2}, kael::DType::F32, kael::Device::CUDA), kael::Error);
}
