#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/data/dataloader.h"
#include "kael/data/dataset.h"
#include "kael/data/mnist.h"
#include "kael/error.h"
#include "kael/kael.h"

using kael::Tensor;
using kael::data::DataLoader;
using kael::data::TensorDataset;

namespace {

class TemporaryFile {
public:
    explicit TemporaryFile(const std::vector<unsigned char>& bytes)
        : path_(std::filesystem::temp_directory_path() /
                ("kael_idx_" +
                 std::string(testing::UnitTest::GetInstance()->current_test_info()->name()))) {
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    ~TemporaryFile() { std::filesystem::remove(path_); }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<unsigned char> images_file() {
    std::vector<unsigned char> bytes = {0, 0, 8, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 2};
    for (unsigned char i = 0; i < 12; ++i) {
        bytes.push_back(static_cast<unsigned char>(i * 20));
    }
    return bytes;
}

std::vector<unsigned char> labels_file() {
    return {0, 0, 8, 1, 0, 0, 0, 3, 7, 0, 9};
}

TensorDataset numbered_dataset(std::int64_t size) {
    const Tensor inputs = kael::arange(0, static_cast<double>(size), 1.0, kael::DType::I64);
    return {inputs, inputs.clone()};
}

} // namespace

TEST(Idx, ReadsImagesAsNormalizedFloats) {
    const TemporaryFile file(images_file());
    const Tensor images = kael::data::read_idx_images(file.path());
    EXPECT_EQ(images.shape(), (kael::Shape{3, 1, 2, 2}));
    EXPECT_EQ(images.dtype(), kael::DType::F32);
    EXPECT_FLOAT_EQ(images.at<float>({0, 0, 0, 1}), 20.0f / 255.0f);
    EXPECT_FLOAT_EQ(images.at<float>({2, 0, 1, 1}), 220.0f / 255.0f);
}

TEST(Idx, ReadsLabels) {
    const TemporaryFile file(labels_file());
    const Tensor labels = kael::data::read_idx_labels(file.path());
    EXPECT_EQ(labels.shape(), (kael::Shape{3}));
    EXPECT_EQ(labels.at<std::int64_t>({0}), 7);
    EXPECT_EQ(labels.at<std::int64_t>({2}), 9);
}

TEST(Idx, BigEndianSizes) {
    std::vector<unsigned char> bytes = {0, 0, 8, 1, 0, 0, 0x01, 0x02};
    bytes.resize(8 + 258, 1);
    const TemporaryFile file(bytes);
    EXPECT_EQ(kael::data::read_idx_labels(file.path()).numel(), 258);
}

TEST(Idx, WrongMagicThrows) {
    const TemporaryFile file(labels_file());
    EXPECT_THROW((void)kael::data::read_idx_images(file.path()), kael::Error);
}

TEST(Idx, TruncatedDataThrows) {
    std::vector<unsigned char> bytes = labels_file();
    bytes.pop_back();
    const TemporaryFile file(bytes);
    EXPECT_THROW((void)kael::data::read_idx_labels(file.path()), kael::Error);
}

TEST(Idx, TrailingDataThrows) {
    std::vector<unsigned char> bytes = labels_file();
    bytes.push_back(0);
    const TemporaryFile file(bytes);
    EXPECT_THROW((void)kael::data::read_idx_labels(file.path()), kael::Error);
}

TEST(Idx, TruncatedHeaderThrows) {
    const TemporaryFile file({0, 0, 8});
    EXPECT_THROW((void)kael::data::read_idx_labels(file.path()), kael::Error);
}

TEST(Idx, HugeDeclaredSizeThrowsBeforeAllocating) {
    const TemporaryFile file(
        {0, 0, 8, 3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF});
    EXPECT_THROW((void)kael::data::read_idx_images(file.path()), kael::Error);
}

TEST(Idx, MissingFileThrows) {
    EXPECT_THROW((void)kael::data::read_idx_labels("/nonexistent/kael/idx"), kael::Error);
}

TEST(TensorDataset, RequiresSameFirstDimension) {
    EXPECT_THROW(TensorDataset(kael::zeros({2, 3}), kael::zeros({3})), kael::Error);
}

TEST(TensorDataset, GetSelectsOneExample) {
    const TensorDataset dataset = numbered_dataset(4);
    EXPECT_EQ(dataset.size(), 4);
    EXPECT_EQ(dataset.get(2).input.item<std::int64_t>(), 2);
}

TEST(Dataset, DefaultBatchStacksExamples) {
    class Squares final : public kael::data::Dataset {
    public:
        [[nodiscard]] std::int64_t size() const override { return 5; }
        [[nodiscard]] kael::data::Example get(std::int64_t i) const override {
            return {kael::full({1}, i * i, kael::DType::I64), kael::full({}, i, kael::DType::I64)};
        }
    };
    const Squares dataset;
    const auto batch = dataset.get_batch(kael::from_vector<std::int64_t>({3, 1}));
    EXPECT_EQ(batch.input.shape(), (kael::Shape{2, 1}));
    EXPECT_EQ(batch.input.at<std::int64_t>({0, 0}), 9);
    EXPECT_EQ(batch.target.at<std::int64_t>({1}), 1);
}

TEST(DataLoader, BatchCounts) {
    const TensorDataset dataset = numbered_dataset(10);
    EXPECT_EQ(DataLoader(dataset, {.batch_size = 3}).num_batches(), 4);
    EXPECT_EQ(DataLoader(dataset, {.batch_size = 3, .drop_last = true}).num_batches(), 3);
    EXPECT_EQ(DataLoader(dataset, {.batch_size = 5}).num_batches(), 2);
}

TEST(DataLoader, RangeForVisitsEveryBatchInOrder) {
    const TensorDataset dataset = numbered_dataset(5);
    const DataLoader loader(dataset, {.batch_size = 2});
    std::vector<std::int64_t> seen;
    std::vector<std::int64_t> sizes;
    for (auto& [input, target] : loader) {
        sizes.push_back(input.numel());
        for (std::int64_t i = 0; i < input.numel(); ++i) {
            seen.push_back(input.at<std::int64_t>({i}));
            EXPECT_EQ(target.at<std::int64_t>({i}), input.at<std::int64_t>({i}));
        }
    }
    EXPECT_EQ(seen, (std::vector<std::int64_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(sizes, (std::vector<std::int64_t>{2, 2, 1}));
}

TEST(DataLoader, DropLastDiscardsIncompleteBatch) {
    const TensorDataset dataset = numbered_dataset(5);
    std::int64_t batches = 0;
    for (const auto& batch : DataLoader(dataset, {.batch_size = 2, .drop_last = true})) {
        EXPECT_EQ(batch.input.numel(), 2);
        ++batches;
    }
    EXPECT_EQ(batches, 2);
}

TEST(DataLoader, ShuffleVisitsEachIndexOncePerEpoch) {
    kael::manual_seed(11);
    const TensorDataset dataset = numbered_dataset(20);
    const DataLoader loader(dataset, {.batch_size = 3, .shuffle = true});
    std::set<std::int64_t> seen;
    std::int64_t count = 0;
    for (const auto& batch : loader) {
        for (std::int64_t i = 0; i < batch.input.numel(); ++i) {
            seen.insert(batch.input.at<std::int64_t>({i}));
            ++count;
        }
    }
    EXPECT_EQ(count, 20);
    EXPECT_EQ(seen.size(), 20u);
}

TEST(DataLoader, ShuffleIsReproducibleWithSeedAndChangesBetweenEpochs) {
    const TensorDataset dataset = numbered_dataset(30);
    const DataLoader loader(dataset, {.batch_size = 30, .shuffle = true});
    const auto epoch_order = [&] {
        std::vector<std::int64_t> order;
        for (const auto& batch : loader) {
            for (std::int64_t i = 0; i < batch.input.numel(); ++i) {
                order.push_back(batch.input.at<std::int64_t>({i}));
            }
        }
        return order;
    };
    kael::manual_seed(12);
    const auto first = epoch_order();
    const auto second = epoch_order();
    kael::manual_seed(12);
    EXPECT_EQ(epoch_order(), first);
    EXPECT_NE(first, second);
}

TEST(DataLoader, InvalidBatchSizeThrows) {
    const TensorDataset dataset = numbered_dataset(2);
    EXPECT_THROW(DataLoader(dataset, {.batch_size = 0}), kael::Error);
}
