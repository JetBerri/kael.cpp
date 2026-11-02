#include <gtest/gtest.h>

#include "kael/kael.h"
#include "kael/nn/loss.h"

using kael::Tensor;

namespace {

const Tensor kMatrix = kael::zeros({2, 3});

} // namespace

TEST(TypedErrors, EveryKindIsAKaelError) {
    static_assert(std::is_base_of_v<kael::Error, kael::IndexError>);
    static_assert(std::is_base_of_v<kael::Error, kael::ValueError>);
    static_assert(std::is_base_of_v<kael::Error, kael::TypeError>);
    SUCCEED();
}

TEST(TypedErrors, IndexProblems) {
    EXPECT_THROW((void)kMatrix.size(2), kael::IndexError);
    EXPECT_THROW((void)kMatrix.select(0, 5), kael::IndexError);
    EXPECT_THROW((void)kMatrix.narrow(0, 9, 1), kael::IndexError);
    EXPECT_THROW((void)kMatrix.at<float>({5, 0}), kael::IndexError);
    EXPECT_THROW((void)kael::index_select(kMatrix, 0, kael::full({1}, 9, kael::DType::I64)),
                 kael::IndexError);
    EXPECT_THROW((void)kael::nn::cross_entropy(kMatrix, kael::full({2}, 7, kael::DType::I64)),
                 kael::IndexError);
}

TEST(TypedErrors, ShapeAndArgumentProblems) {
    EXPECT_THROW((void)kMatrix.view({4}), kael::ValueError);
    EXPECT_THROW((void)kael::zeros({-1}), kael::ValueError);
    EXPECT_THROW((void)kael::add(kMatrix, kael::zeros({4})), kael::ValueError);
    EXPECT_THROW((void)kael::matmul(kMatrix, kMatrix), kael::ValueError);
    EXPECT_THROW((void)kMatrix.item(), kael::ValueError);
    EXPECT_THROW((void)kMatrix.slice(0, 0, 1, 0), kael::ValueError);
    EXPECT_THROW((void)kael::cat({}), kael::ValueError);
    EXPECT_THROW(kael::set_num_threads(0), kael::ValueError);
}

TEST(TypedErrors, DTypeProblems) {
    const Tensor booleans = kael::zeros({2}, kael::DType::Bool);
    const Tensor integers = kael::zeros({2}, kael::DType::I64);
    EXPECT_THROW((void)kael::neg(booleans), kael::TypeError);
    EXPECT_THROW((void)kMatrix.data_ptr<double>(), kael::TypeError);
    EXPECT_THROW((void)kael::mean(integers), kael::TypeError);
    EXPECT_THROW((void)kael::index_select(kMatrix, 0, kael::zeros({1})), kael::TypeError);
    EXPECT_THROW((void)kael::where(integers, 1, 2), kael::TypeError);
}

TEST(TypedErrors, EverythingElseStaysPlain) {
    const Tensor x = kael::zeros({2});
    try {
        x.backward();
        FAIL() << "backward should have thrown";
    } catch (const kael::IndexError&) {
        FAIL() << "not an index problem";
    } catch (const kael::ValueError&) {
        FAIL() << "not a value problem";
    } catch (const kael::TypeError&) {
        FAIL() << "not a type problem";
    } catch (const kael::Error&) {
        SUCCEED();
    }
}
