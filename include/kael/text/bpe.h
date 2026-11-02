#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kael::text {

[[nodiscard]] std::vector<std::string> split_gpt2_words(std::string_view text);

class BpeTokenizer {
public:
    BpeTokenizer(const std::filesystem::path& vocabulary, const std::filesystem::path& merges);

    [[nodiscard]] std::vector<std::int64_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<std::int64_t>& ids) const;
    [[nodiscard]] std::int64_t vocabulary_size() const noexcept {
        return static_cast<std::int64_t>(ids_.size());
    }

    [[nodiscard]] static const std::vector<std::string>& byte_symbols();

private:
    void append_word_ids(std::string_view word, std::vector<std::int64_t>& out) const;

    std::unordered_map<std::string, std::int64_t> ids_;
    std::unordered_map<std::int64_t, std::string> bytes_of_id_;
    std::unordered_map<std::string, std::int64_t> merge_ranks_;
};

} // namespace kael::text
