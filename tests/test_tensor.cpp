#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/tensor.h"

namespace {

// Builds a tensor by hand (there are no factories yet): a storage of
// `storage_elems` elements and contiguous strides unless given.
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
    // Storage of 10 doubles, tensor of 3 starting at element 4.
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
