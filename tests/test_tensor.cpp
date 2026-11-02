#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/factory.h"
#include "kael/tensor.h"

namespace {

kael::Tensor make_tensor(kael::Shape shape, kael::DType dtype = kael::DType::F32,
                         std::int64_t storage_elems = -1, std::int64_t offset = 0,
                         kael::Strides strides = {}) {
    if (storage_elems < 0) {
        storage_elems = kael::numel(shape) + offset;
    }
    auto impl = std::make_shared<kael::TensorImpl>();
    impl->storage = std::make_shared<kael::Storage>(storage_elems * kael::element_size(dtype));
    impl->strides = strides.empty() ? kael::contiguous_strides(shape) : std::move(strides);
    impl->shape = std::move(shape);
    impl->offset = offset;
    impl->dtype = dtype;
    return kael::Tensor::from_impl(std::move(impl));
}

} // namespace

TEST(Tensor, CopySharesData) {
    kael::Tensor a = make_tensor({2, 3});
    kael::Tensor b = a;
    b.data_ptr<float>()[4] = 42.0f;
    EXPECT_EQ(a.data_ptr<float>()[4], 42.0f);
    EXPECT_TRUE(a.is_same(b));
    EXPECT_TRUE(a.shares_storage(b));
}

TEST(Tensor, CloneDoesNotShareData) {
    kael::Tensor a = make_tensor({2, 3});
    for (int i = 0; i < 6; ++i) {
        a.data_ptr<float>()[i] = static_cast<float>(i);
    }
    kael::Tensor c = a.clone();
    EXPECT_FALSE(c.is_same(a));
    EXPECT_FALSE(c.shares_storage(a));
    EXPECT_EQ(c.shape(), a.shape());
    EXPECT_EQ(c.dtype(), a.dtype());
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(c.data_ptr<float>()[i], static_cast<float>(i));
    }
    c.data_ptr<float>()[0] = -1.0f;
    EXPECT_EQ(a.data_ptr<float>()[0], 0.0f);
}

TEST(Tensor, CloneOfOffsetTensorStartsAtZero) {
    kael::Tensor a = make_tensor({3}, kael::DType::I64, 5, 2);
    for (int i = 0; i < 3; ++i) {
        a.data_ptr<std::int64_t>()[i] = 10 + i;
    }
    kael::Tensor c = a.clone();
    EXPECT_EQ(c.offset(), 0);
    EXPECT_EQ(c.impl()->storage->nbytes(), 24);
    EXPECT_EQ(c.data_ptr<std::int64_t>()[2], 12);
}

TEST(Tensor, DataPtrWrongTypeThrows) {
    kael::Tensor a = make_tensor({2}, kael::DType::F32);
    EXPECT_THROW((void)a.data_ptr<double>(), kael::Error);
    EXPECT_THROW((void)a.data_ptr<std::int64_t>(), kael::Error);
    const kael::Tensor& ca = a;
    EXPECT_THROW((void)ca.data_ptr<std::uint8_t>(), kael::Error);
}

TEST(Tensor, DataPtrWrongTypeMessage) {
    kael::Tensor a = make_tensor({2}, kael::DType::F32);
    try {
        (void)a.data_ptr<double>();
        FAIL();
    } catch (const kael::Error& e) {
        EXPECT_NE(std::string(e.what()).find("expected dtype float64 but tensor has float32"),
                  std::string::npos);
    }
}

TEST(Tensor, DataPtrRespectsOffset) {
    kael::Tensor t = make_tensor({3}, kael::DType::F64, 10, 4);
    auto* storage = reinterpret_cast<double*>(t.impl()->storage->data());
    for (int i = 0; i < 10; ++i) {
        storage[i] = static_cast<double>(i);
    }
    EXPECT_EQ(t.data_ptr<double>(), storage + 4);
    EXPECT_EQ(t.data_ptr<double>()[0], 4.0);
    EXPECT_EQ(t.data_ptr<double>()[2], 6.0);
    EXPECT_EQ(t.data_ptr_raw(), t.impl()->storage->data() + 32);
}

TEST(Tensor, Queries) {
    kael::Tensor t = make_tensor({2, 3, 4}, kael::DType::F64);
    EXPECT_EQ(t.shape(), (kael::Shape{2, 3, 4}));
    EXPECT_EQ(t.strides(), (kael::Strides{12, 4, 1}));
    EXPECT_EQ(t.offset(), 0);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.dim(), 3);
    EXPECT_EQ(t.nbytes(), 192);
    EXPECT_EQ(t.dtype(), kael::DType::F64);
    EXPECT_EQ(t.device(), kael::Device::CPU);
    EXPECT_TRUE(t.is_contiguous());
}

TEST(Tensor, ShapeReturnsReference) {
    kael::Tensor t = make_tensor({2, 3});
    EXPECT_EQ(&t.shape(), &t.impl()->shape);
    EXPECT_EQ(&t.strides(), &t.impl()->strides);
}

TEST(Tensor, SizeAndStrideWithNegativeDims) {
    kael::Tensor t = make_tensor({2, 3});
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(-1), 3);
    EXPECT_EQ(t.stride(-1), 1);
    EXPECT_EQ(t.stride(-2), 3);
    EXPECT_THROW((void)t.size(5), kael::Error);
    EXPECT_THROW((void)t.stride(-3), kael::Error);
}

TEST(Tensor, FromImplValidates) {
    auto impl = std::make_shared<kael::TensorImpl>();
    impl->storage = std::make_shared<kael::Storage>(8);
    impl->shape = {3};
    impl->strides = {1};
    EXPECT_THROW(kael::Tensor::from_impl(impl), kael::Error);
    EXPECT_THROW(kael::Tensor::from_impl(nullptr), kael::Error);
}

TEST(Tensor, EmptyTensor) {
    kael::Tensor t = make_tensor({0, 3});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.nbytes(), 0);
    EXPECT_EQ(t.data_ptr<float>(), nullptr);
    kael::Tensor c = t.clone();
    EXPECT_EQ(c.shape(), (kael::Shape{0, 3}));
}

TEST(TensorItem, AnyShapeWithOneElement) {
    EXPECT_EQ(kael::scalar(2.5).item<float>(), 2.5f);
    EXPECT_EQ(kael::full({1}, 3.0, kael::DType::F64).item<double>(), 3.0);
    EXPECT_EQ(kael::full({1, 1}, 4.0, kael::DType::I64).item<std::int64_t>(), 4);
}

TEST(TensorItem, ManyElementsThrows) {
    EXPECT_THROW((void)kael::zeros({2}).item<float>(), kael::Error);
    EXPECT_THROW((void)kael::zeros({2}).item(), kael::Error);
    EXPECT_THROW((void)kael::zeros({0}).item(), kael::Error);
}

TEST(TensorItem, WrongTypeThrows) {
    EXPECT_THROW((void)kael::scalar(1).item<double>(), kael::Error);
}

TEST(TensorItem, UntypedConvertsToDouble) {
    EXPECT_EQ(kael::scalar(1.5).item(), 1.5);
    EXPECT_EQ(kael::scalar(7, kael::DType::I64).item(), 7.0);
    EXPECT_EQ(kael::scalar(1, kael::DType::Bool).item(), 1.0);
}

TEST(TensorAt, ReadsAndWrites) {
    auto t = kael::arange(0, 6, 1, kael::DType::I64);
    EXPECT_EQ(t.at<std::int64_t>({4}), 4);
    t.at<std::int64_t>({4}) = 99;
    EXPECT_EQ(t.data_ptr<std::int64_t>()[4], 99);

    auto m = kael::zeros({2, 3}, kael::DType::F64);
    m.at<double>({1, 2}) = 5.0;
    EXPECT_EQ(m.data_ptr<double>()[5], 5.0);
    const kael::Tensor& cm = m;
    EXPECT_EQ(cm.at<double>({1, 2}), 5.0);
}

TEST(TensorAt, AcceptsSpanAndShape) {
    auto m = kael::zeros({2, 3});
    const kael::Shape index = {1, 0};
    m.at<float>(index) = 1.0f;
    EXPECT_EQ(m.data_ptr<float>()[3], 1.0f);
}

TEST(TensorAt, NegativeIndicesCountFromTheEnd) {
    auto t = kael::arange(0, 6, 1, kael::DType::I64);
    EXPECT_EQ(t.at<std::int64_t>({-1}), 5);
    EXPECT_EQ(t.at<std::int64_t>({-6}), 0);
}

TEST(TensorAt, OutOfRangeThrows) {
    auto m = kael::zeros({2, 3});
    EXPECT_THROW((void)m.at<float>({2, 0}), kael::Error);
    EXPECT_THROW((void)m.at<float>({0, -4}), kael::Error);
    EXPECT_THROW((void)m.at<float>({0}), kael::Error);
    EXPECT_THROW((void)m.at<double>({0, 0}), kael::Error);
}

TEST(TensorAt, RespectsOffsetAndStrides) {
    kael::Tensor t = make_tensor({2}, kael::DType::I64, 10, 3, {4});
    auto* storage = reinterpret_cast<std::int64_t*>(t.impl()->storage->data());
    for (int i = 0; i < 10; ++i) {
        storage[i] = i;
    }
    EXPECT_EQ(t.at<std::int64_t>({0}), 3);
    EXPECT_EQ(t.at<std::int64_t>({1}), 7);
}

TEST(TensorAt, Scalar) {
    auto s = kael::scalar(9, kael::DType::I64);
    EXPECT_EQ(s.at<std::int64_t>({}), 9);
}

namespace {
std::string print(const kael::Tensor& t) {
    std::ostringstream os;
    os << t;
    return os.str();
}
} // namespace

TEST(TensorPrint, Float32) {
    EXPECT_EQ(print(kael::arange(0, 3, 1)), "tensor([0.0000, 1.0000, 2.0000], shape=[3], dtype=float32)");
}

TEST(TensorPrint, Matrix) {
    EXPECT_EQ(print(kael::arange(0, 6, 1, kael::DType::I64).clone()),
              "tensor([0, 1, 2, 3, 4, 5], shape=[6], dtype=int64)");
    EXPECT_EQ(print(kael::from_vector(std::vector<std::int64_t>{0, 1, 2, 3}, {2, 2})),
              "tensor([0, 1, 2, 3], shape=[2, 2], dtype=int64)");
}

TEST(TensorPrint, Bool) {
    EXPECT_EQ(print(kael::from_vector({true, false})),
              "tensor([true, false], shape=[2], dtype=bool)");
}

TEST(TensorPrint, Empty) {
    EXPECT_EQ(print(kael::zeros({0})), "tensor([], shape=[0], dtype=float32)");
}

TEST(TensorPrint, Scalar) {
    EXPECT_EQ(print(kael::scalar(1.5, kael::DType::F64)), "tensor([1.5000], shape=[], dtype=float64)");
}

TEST(TensorPrint, LongTensorIsSummarized) {
    EXPECT_EQ(print(kael::arange(0, 2000, 1, kael::DType::I64)),
              "tensor([0, 1, 2, ..., 1997, 1998, 1999], shape=[2000], dtype=int64)");
}

TEST(TensorPrint, LeavesStreamFlagsUntouched) {
    std::ostringstream os;
    os << kael::scalar(1.0) << " " << 2.5;
    EXPECT_EQ(os.str(), "tensor([1.0000], shape=[], dtype=float32) 2.5");
}

TEST(TensorTo, SameDtypeReturnsSameImpl) {
    auto t = kael::zeros({2});
    EXPECT_TRUE(t.to(kael::DType::F32).is_same(t));
}

TEST(TensorTo, F32ToF64IsExact) {
    auto t = kael::from_vector({0.1f, 2.5f});
    auto d = t.to(kael::DType::F64);
    EXPECT_EQ(d.dtype(), kael::DType::F64);
    EXPECT_EQ(d.data_ptr<double>()[0], static_cast<double>(0.1f));
    EXPECT_FALSE(d.shares_storage(t));
}

TEST(TensorTo, F64ToF32Rounds) {
    auto t = kael::from_vector({0.1});
    EXPECT_EQ(t.to(kael::DType::F32).data_ptr<float>()[0], 0.1f);
}

TEST(TensorTo, FloatToIntTruncatesTowardsZero) {
    auto t = kael::from_vector({2.7, -2.7});
    auto i = t.to(kael::DType::I64);
    EXPECT_EQ(i.data_ptr<std::int64_t>()[0], 2);
    EXPECT_EQ(i.data_ptr<std::int64_t>()[1], -2);
}

TEST(TensorTo, AnythingToBool) {
    auto i = kael::from_vector(std::vector<std::int64_t>{0, 5, -1}).to(kael::DType::Bool);
    EXPECT_EQ(i.data_ptr<std::uint8_t>()[0], 0);
    EXPECT_EQ(i.data_ptr<std::uint8_t>()[1], 1);
    EXPECT_EQ(i.data_ptr<std::uint8_t>()[2], 1);
    auto nan = kael::from_vector({std::nan(""), 0.0}).to(kael::DType::Bool);
    EXPECT_EQ(nan.data_ptr<std::uint8_t>()[0], 1);
    EXPECT_EQ(nan.data_ptr<std::uint8_t>()[1], 0);
}

TEST(TensorTo, BoolToFloat) {
    auto t = kael::from_vector({true, false}).to(kael::DType::F32);
    EXPECT_EQ(t.data_ptr<float>()[0], 1.0f);
    EXPECT_EQ(t.data_ptr<float>()[1], 0.0f);
}

TEST(TensorTo, EmptyAndShapeKept) {
    auto t = kael::zeros({0, 2}).to(kael::DType::I64);
    EXPECT_EQ(t.shape(), (kael::Shape{0, 2}));
    EXPECT_EQ(t.dtype(), kael::DType::I64);
}
