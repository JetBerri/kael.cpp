#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "kael/error.h"
#include "kael/text/bpe.h"

using kael::text::BpeTokenizer;
using kael::text::split_gpt2_words;
using Words = std::vector<std::string>;

TEST(Gpt2Split, SeparatesWordsPunctuationAndLeadingSpaces) {
    EXPECT_EQ(split_gpt2_words("Hello, world!"), (Words{"Hello", ",", " world", "!"}));
}

TEST(Gpt2Split, KeepsContractionsApart) {
    EXPECT_EQ(split_gpt2_words("don't we'll I'm they've you're"),
              (Words{"don", "'t", " we", "'ll", " I", "'m", " they", "'ve", " you", "'re"}));
}

TEST(Gpt2Split, AnUnknownSuffixFallsToPunctuation) {
    EXPECT_EQ(split_gpt2_words("o'clock"), (Words{"o", "'", "clock"}));
}

TEST(Gpt2Split, SeparatesLettersFromNumbers) {
    EXPECT_EQ(split_gpt2_words("abc123 45"), (Words{"abc", "123", " 45"}));
}

TEST(Gpt2Split, TreatsWhitespaceRunsLikeTheReferencePattern) {
    EXPECT_EQ(split_gpt2_words("a  b"), (Words{"a", " ", " b"}));
    EXPECT_EQ(split_gpt2_words("a\n\nb"), (Words{"a", "\n", "\n", "b"}));
    EXPECT_EQ(split_gpt2_words("a   "), (Words{"a", "   "}));
    EXPECT_EQ(split_gpt2_words(" \n x"), (Words{" \n", " x"}));
}

TEST(Gpt2Split, UnderstandsUnicodeClasses) {
    EXPECT_EQ(split_gpt2_words("año €5 日本語"), (Words{"año", " €", "5", " 日本語"}));
}

TEST(Gpt2Split, KeepsEveryByteOfInvalidUtf8) {
    const std::string text = "a\xFF\xC3 b";
    std::string joined;
    for (const std::string& word : split_gpt2_words(text)) {
        joined += word;
    }
    EXPECT_EQ(joined, text);
}

TEST(Gpt2Split, AnEmptyTextHasNoWords) {
    EXPECT_TRUE(split_gpt2_words("").empty());
}

namespace {

class Bpe : public testing::Test {
protected:
    void SetUp() override {
        const std::string name = testing::UnitTest::GetInstance()->current_test_info()->name();
        const auto stem = std::filesystem::temp_directory_path() /
                          ("kael_bpe_" + std::to_string(std::hash<std::string>{}(name)));
        vocabulary_ = stem;
        vocabulary_ += ".json";
        merges_ = stem;
        merges_ += ".txt";
    }
    void TearDown() override {
        std::filesystem::remove(vocabulary_);
        std::filesystem::remove(merges_);
    }

    void write(const std::string& vocabulary, const std::string& merges) const {
        std::ofstream(vocabulary_) << vocabulary;
        std::ofstream(merges_) << merges;
    }

    [[nodiscard]] BpeTokenizer tokenizer() const { return BpeTokenizer(vocabulary_, merges_); }

    static std::string byte_level_vocabulary() {
        nlohmann::json vocabulary = nlohmann::json::object();
        std::int64_t next = 0;
        for (const std::string& symbol : BpeTokenizer::byte_symbols()) {
            vocabulary[symbol] = next++;
        }
        vocabulary["he"] = 100000;
        vocabulary["hel"] = 100001;
        vocabulary["Ġhe"] = 100002;
        vocabulary["<|endoftext|>"] = 100003;
        return vocabulary.dump();
    }

    std::filesystem::path vocabulary_;
    std::filesystem::path merges_;
};

} // namespace

TEST_F(Bpe, MergesFollowTheirRank) {
    write(R"({"h":0,"e":1,"l":2,"he":3,"el":4,"hel":5})", "#version: 0.2\nh e\ne l\nhe l\n");
    EXPECT_EQ(tokenizer().encode("hel"), (std::vector<std::int64_t>{5}));
}

TEST_F(Bpe, ALowerRankWinsOverAnEarlierPosition) {
    write(R"({"h":0,"e":1,"l":2,"he":3,"el":4})", "#version: 0.2\ne l\nh e\n");
    EXPECT_EQ(tokenizer().encode("hel"), (std::vector<std::int64_t>{0, 4}));
}

TEST_F(Bpe, RoundTripsArbitraryBytes) {
    write(byte_level_vocabulary(), "#version: 0.2\nh e\nhe l\nĠ he\n");
    const BpeTokenizer bpe = tokenizer();
    std::string text = "hello he, año\n\n";
    for (int byte = 0; byte < 256; ++byte) {
        text.push_back(static_cast<char>(byte));
    }
    EXPECT_EQ(bpe.decode(bpe.encode(text)), text);
}

TEST_F(Bpe, TheLeadingSpaceUsesItsByteSymbol) {
    write(byte_level_vocabulary(), "#version: 0.2\nh e\nĠ he\n");
    EXPECT_EQ(tokenizer().encode(" he"), (std::vector<std::int64_t>{100002}));
}

TEST_F(Bpe, DecodesSpecialTokensAsText) {
    write(byte_level_vocabulary(), "#version: 0.2\n");
    EXPECT_EQ(tokenizer().decode({100003}), "<|endoftext|>");
}

TEST_F(Bpe, RejectsUnknownIdsWhenDecoding) {
    write(R"({"h":0})", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer().decode({7}), kael::IndexError);
    EXPECT_THROW((void)tokenizer().decode({-1}), kael::IndexError);
}

TEST_F(Bpe, RejectsTextOutsideItsVocabulary) {
    write(R"({"h":0})", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer().encode("x"), kael::ValueError);
}

TEST_F(Bpe, RejectsCorruptFiles) {
    write("not json", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer(), kael::ValueError);
    write(R"({"h":"zero"})", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer(), kael::ValueError);
    write(R"({"h":-1})", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer(), kael::ValueError);
    write(R"({"h":0})", "#version: 0.2\nonlyonesymbol\n");
    EXPECT_THROW((void)tokenizer(), kael::ValueError);
    write(R"({"\u4e00":0})", "#version: 0.2\n");
    EXPECT_THROW((void)tokenizer(), kael::ValueError);
}

TEST_F(Bpe, RejectsMissingFiles) {
    EXPECT_THROW((void)tokenizer(), kael::Error);
}
