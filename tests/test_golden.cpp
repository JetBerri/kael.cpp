#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kael/error.h"
#include "kael/kael.h"
#include "support/allclose.h"
#include "support/npy.h"

using kael::Tensor;

namespace {

const std::filesystem::path kGoldenDir = KAEL_GOLDEN_DIR;

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::istringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

std::vector<std::int64_t> to_ints(const std::vector<std::string>& texts) {
    std::vector<std::int64_t> ints;
    std::ranges::transform(texts, std::back_inserter(ints),
                           [](const std::string& text) { return std::stoll(text); });
    return ints;
}

class Args {
public:
    explicit Args(const std::string& encoded) {
        for (const std::string& item : split(encoded, ',')) {
            const auto colon = item.find(':');
            values_[item.substr(0, colon)] = item.substr(colon + 1);
        }
    }

    [[nodiscard]] std::int64_t integer(const std::string& key) const {
        return std::stoll(values_.at(key));
    }
    [[nodiscard]] double real(const std::string& key) const { return std::stod(values_.at(key)); }
    [[nodiscard]] bool flag(const std::string& key) const {
        return values_.contains(key) && values_.at(key) == "1";
    }
    [[nodiscard]] std::vector<std::int64_t> ints(const std::string& key) const {
        return values_.contains(key) ? to_ints(split(values_.at(key), '/'))
                                     : std::vector<std::int64_t>{};
    }

private:
    std::map<std::string, std::string> values_;
};

Tensor apply_view(const Tensor& t, const std::string& view) {
    const std::vector<std::string> parts = split(view, ':');
    const std::vector<std::int64_t> args =
        to_ints(std::vector<std::string>(parts.begin() + 1, parts.end()));
    if (parts[0] == "none") {
        return t;
    }
    if (parts[0] == "transpose") {
        return t.transpose(args[0], args[1]);
    }
    if (parts[0] == "permute") {
        return t.permute(args);
    }
    if (parts[0] == "slice") {
        return t.slice(args[0], args[1], args[2], args[3]);
    }
    if (parts[0] == "expand") {
        return t.expand(args);
    }
    throw kael::Error("unknown golden view " + view);
}

using Inputs = std::vector<Tensor>;
using GoldenOp = std::function<Tensor(const Inputs&, const Args&)>;

const std::map<std::string, GoldenOp>& golden_ops() {
    static const std::map<std::string, GoldenOp> ops = {
    };
    return ops;
}

struct GoldenCase {
    std::string op;
    std::string name;
    std::vector<std::string> input_names;
    std::map<std::string, std::string> fields;
};

GoldenCase parse_case(const std::string& line) {
    GoldenCase golden;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        const auto equals = token.find('=');
        golden.fields[token.substr(0, equals)] = token.substr(equals + 1);
    }
    golden.op = golden.fields.at("op");
    golden.name = golden.fields.at("case");
    golden.input_names = split(golden.fields.at("inputs"), ',');
    return golden;
}

void run_case(const GoldenCase& golden) {
    const auto op = golden_ops().find(golden.op);
    ASSERT_NE(op, golden_ops().end()) << "no golden op registered for " << golden.op;

    const std::filesystem::path directory = kGoldenDir / golden.op;
    Inputs inputs;
    for (const std::string& input : golden.input_names) {
        const Tensor base = kael::test::read_npy(directory / (golden.name + "_" + input + ".npy"));
        inputs.push_back(apply_view(base, golden.fields.at("view_" + input)));
    }
    const Tensor expected = kael::test::read_npy(directory / (golden.name + "_out.npy"));
    const Tensor actual = op->second(inputs, Args(golden.fields.at("args")));
    EXPECT_TRUE(kael::test::allclose(actual, expected, {.equal_nan = true}));
}

} // namespace

TEST(Golden, AllCasesMatchPyTorch) {
    std::ifstream manifest(kGoldenDir / "manifest.txt");
    ASSERT_TRUE(manifest.is_open());

    std::string line;
    while (std::getline(manifest, line)) {
        const GoldenCase golden = parse_case(line);
        SCOPED_TRACE(golden.op + "/" + golden.name);
        try {
            run_case(golden);
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
}
