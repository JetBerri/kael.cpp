#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "kael/kael.h"
#include "kael/matmul_versions.h"
#include "kael/parallel.h"

#include "support/allclose.h"

using kael::Tensor;
using kael::matmul_versions::Sizes;

namespace {

template <typename T> using Version = void (*)(const T*, const T*, T*, const Sizes&);

template <typename T> std::vector<std::pair<std::string, Version<T>>> versions() {
    return {{"v2", kael::matmul_versions::v2_ikj<T>},
            {"v3", kael::matmul_versions::v3_tiled<T>},
            {"v4", kael::matmul_versions::v4_packed<T>},
            {"v5", kael::matmul_versions::v5_simd<T>},
            {"v6", kael::matmul_versions::v6_threaded<T>},
            {"multiply", kael::matmul_versions::multiply<T>}};
}

template <typename T> kael::DType dtype_of() {
    return kael::dtype_of_v<T>;
}

template <typename T> void expect_all_versions_match(const Sizes& sizes) {
    const Tensor a = kael::randn({sizes.rows, sizes.inner}, dtype_of<T>());
    const Tensor b = kael::randn({sizes.inner, sizes.columns}, dtype_of<T>());
    Tensor reference = kael::full({sizes.rows, sizes.columns}, 1, dtype_of<T>());
    kael::matmul_versions::v1_ijk<T>(a.data_ptr<T>(), b.data_ptr<T>(), reference.data_ptr<T>(),
                                     sizes);

    const auto tolerance = std::is_same_v<T, float> ? kael::test::Tolerance{1e-3, 1e-3}
                                                    : kael::test::Tolerance{1e-11, 1e-11};
    for (const auto& [name, version] : versions<T>()) {
        Tensor out = kael::full({sizes.rows, sizes.columns}, 1, dtype_of<T>());
        version(a.data_ptr<T>(), b.data_ptr<T>(), out.data_ptr<T>(), sizes);
        EXPECT_TRUE(kael::test::allclose(out, reference, {.tolerance = tolerance}))
            << name << " " << sizes.rows << "x" << sizes.inner << "x" << sizes.columns;
    }
}

} // namespace

class MatmulVersions : public testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatmulVersions, MatchReferenceInBothPrecisions) {
    const auto [rows, inner, columns] = GetParam();
    kael::manual_seed(static_cast<std::uint64_t>(rows * 1000000 + inner * 1000 + columns));
    const Sizes sizes{rows, inner, columns};
    expect_all_versions_match<float>(sizes);
    expect_all_versions_match<double>(sizes);
}

INSTANTIATE_TEST_SUITE_P(Shapes, MatmulVersions,
                         testing::Values(std::tuple{1, 1, 1}, std::tuple{1, 7, 1},
                                         std::tuple{7, 1, 13}, std::tuple{8, 16, 12},
                                         std::tuple{9, 5, 13}, std::tuple{6, 8, 8},
                                         std::tuple{7, 9, 9}, std::tuple{33, 40, 25},
                                         std::tuple{100, 3, 100}, std::tuple{65, 257, 31},
                                         std::tuple{385, 513, 13}, std::tuple{13, 513, 97},
                                         std::tuple{200, 200, 200}));

TEST(MatmulVersionsEmpty, ZeroSizesLeaveOutputUntouched) {
    float value = 5.0f;
    const Sizes sizes{1, 0, 1};
    kael::matmul_versions::v5_simd<float>(nullptr, nullptr, &value, sizes);
    kael::matmul_versions::v6_threaded<float>(nullptr, nullptr, &value, sizes);
    EXPECT_EQ(value, 5.0f);
}

TEST(MatmulVersionsThreads, ResultDoesNotDependOnThreadCount) {
    const std::int64_t previous = kael::get_num_threads();
    kael::manual_seed(77);
    const Tensor a = kael::randn({300, 300});
    const Tensor b = kael::randn({300, 300});

    kael::set_num_threads(1);
    const Tensor single = kael::matmul(a, b);
    kael::set_num_threads(4);
    const Tensor several = kael::matmul(a, b);
    kael::set_num_threads(previous);

    EXPECT_TRUE(kael::equal(single, several));
}

TEST(MatmulDispatch, LargeIntegerMatrixStillUsesExactArithmetic) {
    const Tensor a = kael::full({50, 50}, 3, kael::DType::I64);
    const Tensor b = kael::full({50, 50}, 2, kael::DType::I64);
    EXPECT_EQ(kael::matmul(a, b).at<std::int64_t>({49, 49}), 300);
}
