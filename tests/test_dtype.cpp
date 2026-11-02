#include <cstdint>
#include <sstream>
#include <type_traits>

#include <gtest/gtest.h>

#include "kael/dtype.h"
#include "kael/error.h"

using kael::Device;
using kael::DType;

namespace {
constexpr DType kAll[] = {DType::F32, DType::F64, DType::I64, DType::Bool};
}

TEST(DType, ElementSize) {
    EXPECT_EQ(kael::element_size(DType::F32), 4);
    EXPECT_EQ(kael::element_size(DType::F64), 8);
    EXPECT_EQ(kael::element_size(DType::I64), 8);
    EXPECT_EQ(kael::element_size(DType::Bool), 1);
}

TEST(DType, ToString) {
    EXPECT_STREQ(kael::to_string(DType::F32), "float32");
    EXPECT_STREQ(kael::to_string(DType::F64), "float64");
    EXPECT_STREQ(kael::to_string(DType::I64), "int64");
    EXPECT_STREQ(kael::to_string(DType::Bool), "bool");
}

TEST(DType, StreamOperator) {
    std::ostringstream os;
    os << DType::F64 << " " << Device::CPU << " " << Device::CUDA;
    EXPECT_EQ(os.str(), "float64 cpu cuda");
}

TEST(DType, IsFloatingPoint) {
    EXPECT_TRUE(kael::is_floating_point(DType::F32));
    EXPECT_TRUE(kael::is_floating_point(DType::F64));
    EXPECT_FALSE(kael::is_floating_point(DType::I64));
    EXPECT_FALSE(kael::is_floating_point(DType::Bool));
}

TEST(DType, TraitCppToDType) {
    static_assert(kael::dtype_of_v<float> == DType::F32);
    static_assert(kael::dtype_of_v<double> == DType::F64);
    static_assert(kael::dtype_of_v<std::int64_t> == DType::I64);
    static_assert(kael::dtype_of_v<std::uint8_t> == DType::Bool);
    static_assert(std::is_same_v<kael::cpp_type_t<DType::F32>, float>);
    static_assert(std::is_same_v<kael::cpp_type_t<DType::Bool>, std::uint8_t>);
}

TEST(DType, DispatchMatchesElementSize) {
    for (DType d : kAll) {
        const auto bytes =
            kael::dispatch(d, []<typename T>() { return static_cast<std::int64_t>(sizeof(T)); });
        EXPECT_EQ(bytes, kael::element_size(d)) << d;
    }
}

TEST(DType, DispatchSelectsMatchingType) {
    for (DType d : kAll) {
        const DType seen = kael::dispatch(d, []<typename T>() { return kael::dtype_of_v<T>; });
        EXPECT_EQ(seen, d);
    }
}

TEST(DType, DispatchVoidLambda) {
    int calls = 0;
    kael::dispatch(DType::I64, [&]<typename T>() { ++calls; });
    EXPECT_EQ(calls, 1);
}

TEST(DType, DispatchFloatingRejectsIntegers) {
    auto f = []<typename T>() { return sizeof(T); };
    EXPECT_EQ(kael::dispatch_floating(DType::F32, "test", f), sizeof(float));
    EXPECT_EQ(kael::dispatch_floating(DType::F64, "test", f), sizeof(double));
    EXPECT_THROW(kael::dispatch_floating(DType::I64, "test", f), kael::Error);
    EXPECT_THROW(kael::dispatch_floating(DType::Bool, "test", f), kael::Error);
}

TEST(DType, DispatchNoBoolRejectsBool) {
    auto f = []<typename T>() { return kael::dtype_of_v<T>; };
    EXPECT_EQ(kael::dispatch_no_bool(DType::I64, "test", f), DType::I64);
    EXPECT_THROW(kael::dispatch_no_bool(DType::Bool, "test", f), kael::Error);
}
