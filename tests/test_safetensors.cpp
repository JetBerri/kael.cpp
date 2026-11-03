#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include "kael/io/safetensors.h"
#include "kael/kael.h"
#include "kael/nn/linear.h"

using kael::DType;
using kael::Tensor;
using kael::io::TensorMap;
namespace io = kael::io;

namespace {

class Safetensors : public testing::Test {
protected:
    void SetUp() override {
        const std::string name = testing::UnitTest::GetInstance()->current_test_info()->name();
        path_ = std::filesystem::temp_directory_path() /
                ("kael_test_" + std::to_string(std::hash<std::string>{}(name)) + ".safetensors");
    }
    void TearDown() override { std::filesystem::remove(path_); }

    void write_raw(const std::string& header, const std::string& data = "") const {
        std::ofstream file(path_, std::ios::binary);
        const std::uint64_t size = header.size();
        char bytes[8];
        std::memcpy(bytes, &size, 8);
        file.write(bytes, 8);
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    std::filesystem::path path_;
};

} // namespace

TEST_F(Safetensors, RoundTripEveryDtype) {
    kael::manual_seed(1);
    const std::map<std::string, Tensor> tensors = {
        {"a", kael::randn({2, 3})},
        {"b.c", kael::randn({4}, DType::F64)},
        {"integers", kael::randint(-5, 5, {3, 1})},
        {"flags", kael::randn({5}) > 0},
        {"scalar", kael::full({}, 2.5)},
        {"empty", kael::zeros({0, 3})},
    };
    io::save_safetensors(path_, tensors);
    const auto loaded = io::load_safetensors(path_);
    ASSERT_EQ(loaded.size(), tensors.size());
    for (const auto& [name, tensor] : tensors) {
        EXPECT_EQ(loaded.at(name).dtype(), tensor.dtype()) << name;
        EXPECT_TRUE(kael::equal(loaded.at(name), tensor)) << name;
    }
}

TEST_F(Safetensors, SavesViewsByValue) {
    const Tensor x = kael::arange(0, 6, 1).reshape({2, 3});
    io::save_safetensors(path_, {{"view", x.T()}});
    EXPECT_TRUE(kael::equal(io::load_safetensors(path_).at("view"), x.T().contiguous()));
}

TEST_F(Safetensors, MetadataRoundTrip) {
    io::save_safetensors(path_, {{"a", kael::ones({1})}}, {{"format", "pt"}, {"note", "hello"}});
    const auto file = io::read_safetensors(path_);
    EXPECT_EQ(file.metadata.at("format"), "pt");
    EXPECT_EQ(file.metadata.at("note"), "hello");
    EXPECT_EQ(file.tensors.size(), 1u);
}

TEST_F(Safetensors, HeaderFormatMatchesTheSpecification) {
    io::save_safetensors(path_, {{"layer", kael::ones({2, 2})}});
    std::ifstream file(path_, std::ios::binary);
    std::uint64_t size = 0;
    file.read(reinterpret_cast<char*>(&size), 8);
    std::string header(size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(size));
    EXPECT_NE(header.find("\"dtype\":\"F32\""), std::string::npos);
    EXPECT_NE(header.find("\"shape\":[2,2]"), std::string::npos);
    EXPECT_NE(header.find("\"data_offsets\":[0,16]"), std::string::npos);
    EXPECT_EQ((8 + size) % 8, 0u);
}

TEST_F(Safetensors, ReadsAFileWithPadding) {
    write_raw(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}     )",
              std::string("\x00\x00\x80\x3f", 4));
    EXPECT_EQ(io::load_safetensors(path_).at("a").item<float>(), 1.0f);
}

TEST_F(Safetensors, StateDictRoundTrip) {
    kael::manual_seed(2);
    kael::nn::Linear first(3, 2);
    kael::nn::Linear second(3, 2);
    io::save_safetensors(path_, first.state_dict());
    second.load_state_dict(io::load_safetensors(path_));
    EXPECT_TRUE(kael::equal(first.weight().detach(), second.weight().detach()));
}

TEST_F(Safetensors, MissingFile) {
    EXPECT_THROW((void)io::load_safetensors("/nonexistent/file"), kael::Error);
}

TEST_F(Safetensors, TruncatedLengthPrefix) {
    std::ofstream(path_, std::ios::binary) << "abc";
    EXPECT_THROW((void)io::load_safetensors(path_), kael::ValueError);
}

TEST_F(Safetensors, HeaderLongerThanFile) {
    std::ofstream file(path_, std::ios::binary);
    const std::uint64_t size = 1000;
    file.write(reinterpret_cast<const char*>(&size), 8);
    file.close();
    EXPECT_THROW((void)io::load_safetensors(path_), kael::ValueError);
}

TEST_F(Safetensors, HugeHeaderLengthDoesNotOverflow) {
    std::ofstream file(path_, std::ios::binary);
    const std::uint64_t size = 0xFFFFFFFFFFFFFFFFull;
    file.write(reinterpret_cast<const char*>(&size), 8);
    file.close();
    EXPECT_THROW((void)io::load_safetensors(path_), kael::ValueError);
}

class BadHeader : public Safetensors, public testing::WithParamInterface<const char*> {};

TEST_P(BadHeader, IsRejected) {
    write_raw(GetParam(), std::string(16, '\0'));
    EXPECT_THROW((void)io::load_safetensors(path_), kael::ValueError) << GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    Corrupt, BadHeader,
    testing::Values(
        "not json", "[]", "3", R"({"a":1})", R"({"a":{"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","data_offsets":[0,4]}})", R"({"a":{"dtype":"F32","shape":[1]}})",
        R"({"a":{"dtype":7,"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":1,"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":["1"],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})",
        R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,32]}})",
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"b":{"dtype":"F32","shape":[3],"data_offsets":[4,16]}})",
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"c":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}})",
        R"({"a":{"dtype":"F32","shape":[9223372036854775807,9223372036854775807],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,18446744073709551615]}})",
        R"({"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
        R"({"__metadata__":3,"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})"));

TEST_F(Safetensors, InvalidBoolBytesAreRejected) {
    write_raw(R"({"a":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}})", std::string("\x02", 1));
    EXPECT_THROW((void)io::load_safetensors(path_), kael::ValueError);
}

TEST_F(Safetensors, UnsupportedDtypeIsATypeError) {
    write_raw(R"({"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}})", std::string(2, '\0'));
    EXPECT_THROW((void)io::load_safetensors(path_), kael::TypeError);
}

TEST_F(Safetensors, NamesCannotBeTheMetadataKey) {
    EXPECT_THROW(io::save_safetensors(path_, {{"__metadata__", kael::ones({1})}}),
                 kael::ValueError);
}

TEST_F(Safetensors, MappedLoadMatchesTheCopyingLoad) {
    kael::manual_seed(7);
    const TensorMap tensors = {{"a", kael::randn({20, 30})},
                               {"b", kael::randn({5}, DType::F64)},
                               {"c", kael::randint(0, 9, {4, 2})},
                               {"d", kael::from_vector<std::uint8_t>({1, 0, 1}, {3}).to(DType::Bool)},
                               {"empty", kael::zeros({0, 3})}};
    kael::io::save_safetensors(path_, tensors);

    const TensorMap copied = kael::io::load_safetensors(path_);
    const TensorMap mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped);
    ASSERT_EQ(mapped.size(), copied.size());
    for (const auto& [name, tensor] : copied) {
        EXPECT_EQ(mapped.at(name).dtype(), tensor.dtype()) << name;
        EXPECT_EQ(mapped.at(name).shape(), tensor.shape()) << name;
        EXPECT_TRUE(kael::equal(mapped.at(name), tensor)) << name;
    }
}

TEST_F(Safetensors, MappedTensorsOutliveTheMap) {
    kael::io::save_safetensors(path_, {{"a", kael::full({64, 64}, 3.0)}, {"b", kael::zeros({2})}});
    kael::Tensor kept = kael::zeros({1});
    {
        const TensorMap mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped);
        kept = mapped.at("a");
    }
    std::filesystem::remove(path_);
    EXPECT_EQ(kael::sum(kept).item<float>(), 3.0f * 64.0f * 64.0f);
}

TEST_F(Safetensors, WritingToAMappedTensorDoesNotChangeTheFile) {
    kael::io::save_safetensors(path_, {{"a", kael::full({4}, 1.0)}});
    {
        kael::Tensor mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped).at("a");
        mapped.fill_(9.0);
        EXPECT_EQ(mapped.at<float>({0}), 9.0f);
    }
    EXPECT_EQ(kael::io::load_safetensors(path_).at("a").at<float>({0}), 1.0f);
}

TEST_F(Safetensors, MappedLoadSharesTheFilePagesWhenAligned) {
    kael::io::save_safetensors(path_, {{"a", kael::full({4}, 1.0)}, {"b", kael::full({4}, 2.0)}});
    const TensorMap mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped);
    const auto gap = mapped.at("b").data_ptr<float>() - mapped.at("a").data_ptr<float>();
    EXPECT_EQ(gap, 4);
}

TEST_F(Safetensors, MappedLoadCopiesMisalignedTensors) {
    const std::string text =
        R"({"pad":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"F64","shape":[1],"data_offsets":[1,9]}})";
    const std::string header = text + std::string((8 - text.size() % 8) % 8, ' ');
    const double value = 2.5;
    std::string data(1, '\1');
    data.append(reinterpret_cast<const char*>(&value), sizeof(value));
    write_raw(header, data);
    const TensorMap mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped);
    const kael::Tensor a = mapped.at("a");
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data_ptr<double>()) % alignof(double), 0u);
    EXPECT_EQ(a.item<double>(), 2.5);
}

TEST_F(Safetensors, MappedLoadRejectsCorruptFilesLikeTheCopyingLoad) {
    write_raw("nope");
    EXPECT_THROW((void)kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped), kael::Error);
}

TEST_F(Safetensors, SavingOverTheFileBehindAMappedTensorKeepsItValid) {
    kael::manual_seed(8);
    const kael::Tensor original = kael::randn({512, 512});
    kael::io::save_safetensors(path_, {{"a", original}});
    const TensorMap mapped = kael::io::load_safetensors(path_, kael::io::LoadMode::Mapped);

    kael::io::save_safetensors(path_, mapped);

    EXPECT_TRUE(kael::equal(mapped.at("a"), original));
    EXPECT_TRUE(kael::equal(kael::io::load_safetensors(path_).at("a"), original));
}

TEST_F(Safetensors, AFailedSaveLeavesTheExistingFileIntact) {
    kael::io::save_safetensors(path_, {{"a", kael::full({2}, 1.0)}});
    const TensorMap invalid = {{"__metadata__", kael::zeros({1})}};
    EXPECT_THROW(kael::io::save_safetensors(path_, invalid), kael::Error);
    EXPECT_EQ(kael::io::load_safetensors(path_).at("a").at<float>({0}), 1.0f);
}

TEST_F(Safetensors, SavingLeavesNoTemporaryFilesBehind) {
    kael::io::save_safetensors(path_, {{"a", kael::full({2}, 1.0)}});
    int siblings = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path_.parent_path())) {
        if (entry.path().filename().string().starts_with(path_.filename().string())) {
            ++siblings;
        }
    }
    EXPECT_EQ(siblings, 1);
}
