#include "kael/text/bpe.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <sstream>

#include "kael/error.h"

namespace kael::text {

namespace {

struct CodePointRange {
    char32_t first;
    char32_t last;
};

#include "unicode_classes.inc"

constexpr std::array<CodePointRange, 10> kWhitespace = {{{0x9, 0xD},
                                                         {0x20, 0x20},
                                                         {0x85, 0x85},
                                                         {0xA0, 0xA0},
                                                         {0x1680, 0x1680},
                                                         {0x2000, 0x200A},
                                                         {0x2028, 0x2029},
                                                         {0x202F, 0x202F},
                                                         {0x205F, 0x205F},
                                                         {0x3000, 0x3000}}};

constexpr char32_t kInvalidCodePoint = 0xFFFD;
constexpr char32_t kFirstUnprintableSymbol = 256;

bool contains(std::span<const CodePointRange> ranges, char32_t code_point) {
    const auto* found = std::upper_bound(
        ranges.data(), ranges.data() + ranges.size(), code_point,
        [](char32_t value, const CodePointRange& range) { return value < range.first; });
    return found != ranges.data() && code_point <= (found - 1)->last;
}

bool is_letter(char32_t code_point) {
    return contains(kLetters, code_point);
}
bool is_number(char32_t code_point) {
    return contains(kNumbers, code_point);
}
bool is_space(char32_t code_point) {
    return contains(kWhitespace, code_point);
}
bool is_other(char32_t code_point) {
    return !is_space(code_point) && !is_letter(code_point) && !is_number(code_point);
}

struct DecodedPoint {
    char32_t code_point;
    std::size_t size;
};

DecodedPoint decode_utf8(std::string_view text, std::size_t position) {
    const auto byte_at = [&](std::size_t i) { return static_cast<unsigned char>(text[i]); };
    const unsigned char lead = byte_at(position);
    if (lead < 0x80) {
        return {lead, 1};
    }
    std::size_t size = 0;
    char32_t value = 0;
    char32_t smallest = 0;
    if ((lead & 0xE0) == 0xC0) {
        size = 2;
        value = lead & 0x1Fu;
        smallest = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
        size = 3;
        value = lead & 0x0Fu;
        smallest = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
        size = 4;
        value = lead & 0x07u;
        smallest = 0x10000;
    } else {
        return {kInvalidCodePoint, 1};
    }
    if (position + size > text.size()) {
        return {kInvalidCodePoint, 1};
    }
    for (std::size_t i = 1; i < size; ++i) {
        const unsigned char next = byte_at(position + i);
        if ((next & 0xC0) != 0x80) {
            return {kInvalidCodePoint, 1};
        }
        value = (value << 6) | (next & 0x3Fu);
    }
    const bool surrogate = value >= 0xD800 && value <= 0xDFFF;
    if (value < smallest || value > 0x10FFFF || surrogate) {
        return {kInvalidCodePoint, 1};
    }
    return {value, size};
}

std::string encode_utf8(char32_t code_point) {
    std::string out;
    if (code_point < 0x80) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
    return out;
}

class WordSplitter {
public:
    explicit WordSplitter(std::string_view text) : text_(text) {
        for (std::size_t position = 0; position < text.size();) {
            const DecodedPoint point = decode_utf8(text, position);
            points_.push_back(point.code_point);
            offsets_.push_back(position);
            position += point.size;
        }
        offsets_.push_back(text.size());
    }

    [[nodiscard]] std::vector<std::string> split() const {
        std::vector<std::string> words;
        for (std::size_t begin = 0; begin < points_.size();) {
            const std::size_t end = word_end(begin);
            words.emplace_back(text_.substr(offsets_[begin], offsets_[end] - offsets_[begin]));
            begin = end;
        }
        return words;
    }

private:
    [[nodiscard]] char32_t point_or_null(std::size_t index) const {
        return index < points_.size() ? points_[index] : U'\0';
    }

    [[nodiscard]] std::size_t contraction_end(std::size_t begin) const {
        if (points_[begin] != U'\'') {
            return begin;
        }
        const char32_t first = point_or_null(begin + 1);
        const char32_t second = point_or_null(begin + 2);
        const bool pair = (first == U'l' && second == U'l') || (first == U'v' && second == U'e') ||
                          (first == U'r' && second == U'e');
        if (pair) {
            return begin + 3;
        }
        const bool single = first == U's' || first == U'd' || first == U'm' || first == U't';
        return single ? begin + 2 : begin;
    }

    template <typename Predicate>
    [[nodiscard]] std::size_t class_run_end(std::size_t begin, Predicate belongs) const {
        const std::size_t start = points_[begin] == U' ' ? begin + 1 : begin;
        if (start >= points_.size() || !belongs(points_[start])) {
            return begin;
        }
        std::size_t end = start;
        while (end < points_.size() && belongs(points_[end])) {
            ++end;
        }
        return end;
    }

    [[nodiscard]] std::size_t whitespace_end(std::size_t begin) const {
        std::size_t end = begin;
        while (end < points_.size() && is_space(points_[end])) {
            ++end;
        }
        if (end == points_.size() || end - begin == 1) {
            return end;
        }
        return end - 1;
    }

    [[nodiscard]] std::size_t word_end(std::size_t begin) const {
        for (const std::size_t end :
             {contraction_end(begin), class_run_end(begin, is_letter),
              class_run_end(begin, is_number), class_run_end(begin, is_other)}) {
            if (end != begin) {
                return end;
            }
        }
        return whitespace_end(begin);
    }

    std::string_view text_;
    std::vector<char32_t> points_;
    std::vector<std::size_t> offsets_;
};

std::array<char32_t, 256> make_symbol_of_byte() {
    std::array<char32_t, 256> symbols{};
    char32_t next_unprintable = kFirstUnprintableSymbol;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool printable = (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) ||
                               (byte >= 0xAE && byte <= 0xFF);
        symbols[byte] = printable ? static_cast<char32_t>(byte) : next_unprintable++;
    }
    return symbols;
}

const std::array<char32_t, 256>& symbol_of_byte() {
    static const std::array<char32_t, 256> symbols = make_symbol_of_byte();
    return symbols;
}

const std::unordered_map<char32_t, char>& byte_of_symbol() {
    static const std::unordered_map<char32_t, char> bytes = [] {
        std::unordered_map<char32_t, char> reverse;
        for (unsigned byte = 0; byte < 256; ++byte) {
            reverse.emplace(symbol_of_byte()[byte], static_cast<char>(byte));
        }
        return reverse;
    }();
    return bytes;
}

std::string bytes_of_token(const std::string& token) {
    std::string bytes;
    for (std::size_t position = 0; position < token.size();) {
        const DecodedPoint point = decode_utf8(token, position);
        const auto found = byte_of_symbol().find(point.code_point);
        KAEL_CHECK_VALUE(found != byte_of_symbol().end(),
                         "vocabulary token '" << token << "' is not byte-level BPE");
        bytes.push_back(found->second);
        position += point.size;
    }
    return bytes;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    KAEL_CHECK(file.is_open(), "cannot open tokenizer file " << path);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left).push_back(' ');
    key.append(right);
    return key;
}

} // namespace

std::vector<std::string> split_gpt2_words(std::string_view text) {
    return WordSplitter(text).split();
}

const std::vector<std::string>& BpeTokenizer::byte_symbols() {
    static const std::vector<std::string> symbols = [] {
        std::vector<std::string> all;
        all.reserve(256);
        for (const char32_t symbol : symbol_of_byte()) {
            all.push_back(encode_utf8(symbol));
        }
        return all;
    }();
    return symbols;
}

BpeTokenizer::BpeTokenizer(const std::filesystem::path& vocabulary,
                           const std::filesystem::path& merges) {
    const auto parsed = nlohmann::json::parse(read_file(vocabulary), nullptr, false);
    KAEL_CHECK_VALUE(!parsed.is_discarded() && parsed.is_object(),
                     "vocabulary " << vocabulary << " is not a JSON object");
    for (const auto& [token, id] : parsed.items()) {
        KAEL_CHECK_VALUE(id.is_number_integer() && id.get<std::int64_t>() >= 0,
                         "vocabulary token '" << token << "' has an invalid id");
        const auto value = id.get<std::int64_t>();
        ids_.emplace(token, value);
        bytes_of_id_.emplace(value, bytes_of_token(token));
    }

    std::istringstream lines(read_file(merges));
    std::string line;
    std::int64_t rank = 0;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.starts_with("#version")) {
            continue;
        }
        const std::size_t space = line.find(' ');
        KAEL_CHECK_VALUE(space != std::string::npos && space > 0 && space + 1 < line.size() &&
                             line.find(' ', space + 1) == std::string::npos,
                         "merges line '" << line << "' is not a pair of symbols");
        merge_ranks_.emplace(line, rank++);
    }
}

void BpeTokenizer::append_word_ids(std::string_view word, std::vector<std::int64_t>& out) const {
    std::vector<std::string> symbols;
    symbols.reserve(word.size());
    for (const char byte : word) {
        symbols.push_back(byte_symbols()[static_cast<unsigned char>(byte)]);
    }

    while (symbols.size() > 1) {
        std::int64_t best_rank = std::numeric_limits<std::int64_t>::max();
        std::size_t best = symbols.size();
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto found = merge_ranks_.find(merge_key(symbols[i], symbols[i + 1]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best = i;
            }
        }
        if (best == symbols.size()) {
            break;
        }
        const std::string left = symbols[best];
        const std::string right = symbols[best + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(std::move(symbols[i]));
            }
        }
        symbols = std::move(merged);
    }

    for (const std::string& symbol : symbols) {
        const auto found = ids_.find(symbol);
        KAEL_CHECK_VALUE(found != ids_.end(), "token '" << symbol << "' is not in the vocabulary");
        out.push_back(found->second);
    }
}

std::vector<std::int64_t> BpeTokenizer::encode(std::string_view text) const {
    std::vector<std::int64_t> ids;
    for (const std::string& word : split_gpt2_words(text)) {
        append_word_ids(word, ids);
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<std::int64_t>& ids) const {
    std::string text;
    for (const std::int64_t id : ids) {
        const auto found = bytes_of_id_.find(id);
        KAEL_CHECK_INDEX(found != bytes_of_id_.end(), "token id " << id << " is not in the vocabulary");
        text += found->second;
    }
    return text;
}

} // namespace kael::text
