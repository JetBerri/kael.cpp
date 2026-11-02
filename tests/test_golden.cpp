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
#include "kael/ops.h"
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

    [[nodiscard]] bool has(const std::string& key) const { return values_.contains(key); }
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

GoldenOp binary(Tensor (*op)(const kael::Operand&, const kael::Operand&)) {
    return [op](const Inputs& inputs, const Args&) { return op(inputs[0], inputs[1]); };
}

GoldenOp inplace(Tensor& (Tensor::*op)(const kael::Operand&)) {
    return [op](const Inputs& inputs, const Args&) {
        Tensor self = inputs[0];
        return (self.*op)(inputs[1]);
    };
}

GoldenOp reduction(Tensor (*op)(const Tensor&, const kael::Dims&, bool)) {
    return [op](const Inputs& inputs, const Args& args) {
        return op(inputs[0], args.ints("dims"), args.flag("keepdim"));
    };
}

GoldenOp reduction_with_correction(Tensor (*op)(const Tensor&, const kael::Dims&, bool, std::int64_t),
                                   std::int64_t correction) {
    return [op, correction](const Inputs& inputs, const Args& args) {
        return op(inputs[0], args.ints("dims"), args.flag("keepdim"), correction);
    };
}

GoldenOp arg_reduction(Tensor (*flat)(const Tensor&), Tensor (*along)(const Tensor&, std::int64_t, bool)) {
    return [flat, along](const Inputs& inputs, const Args& args) {
        return args.has("dim") ? along(inputs[0], args.integer("dim"), args.flag("keepdim"))
                               : flat(inputs[0]);
    };
}

GoldenOp tensor_pair(Tensor (*op)(const Tensor&, const Tensor&)) {
    return [op](const Inputs& inputs, const Args&) { return op(inputs[0], inputs[1]); };
}

GoldenOp unary(Tensor (*op)(const Tensor&)) {
    return [op](const Inputs& inputs, const Args&) { return op(inputs[0]); };
}

const std::map<std::string, GoldenOp>& golden_ops() {
    static const std::map<std::string, GoldenOp> ops = {
        {"add", binary(kael::add)},
        {"sub", binary(kael::sub)},
        {"mul", binary(kael::mul)},
        {"div", binary(kael::div)},
        {"floor_divide", binary(kael::floor_divide)},
        {"pow", binary(kael::pow)},
        {"maximum", binary(kael::maximum)},
        {"minimum", binary(kael::minimum)},
        {"eq", binary(kael::eq)},
        {"ne", binary(kael::ne)},
        {"lt", binary(kael::lt)},
        {"le", binary(kael::le)},
        {"gt", binary(kael::gt)},
        {"ge", binary(kael::ge)},
        {"logical_and", binary(kael::logical_and)},
        {"logical_or", binary(kael::logical_or)},
        {"logical_xor", binary(kael::logical_xor)},
        {"neg", unary(kael::neg)},
        {"abs", unary(kael::abs)},
        {"sign", unary(kael::sign)},
        {"exp", unary(kael::exp)},
        {"log", unary(kael::log)},
        {"log1p", unary(kael::log1p)},
        {"expm1", unary(kael::expm1)},
        {"sqrt", unary(kael::sqrt)},
        {"rsqrt", unary(kael::rsqrt)},
        {"sin", unary(kael::sin)},
        {"cos", unary(kael::cos)},
        {"tanh", unary(kael::tanh)},
        {"sigmoid", unary(kael::sigmoid)},
        {"relu", unary(kael::relu)},
        {"gelu", unary(kael::gelu)},
        {"add_", inplace(&Tensor::add_)},
        {"sub_", inplace(&Tensor::sub_)},
        {"mul_", inplace(&Tensor::mul_)},
        {"div_", inplace(&Tensor::div_)},
        {"pow_", inplace(&Tensor::pow_)},
        {"copy_",
         [](const Inputs& inputs, const Args&) {
             Tensor self = inputs[0];
             return self.copy_(inputs[1]);
         }},
        {"clamp_",
         [](const Inputs& inputs, const Args& args) {
             Tensor self = inputs[0];
             return self.clamp_(args.real("min"), args.real("max"));
         }},
        {"sum", reduction(kael::sum)},
        {"prod", reduction(kael::prod)},
        {"mean", reduction(kael::mean)},
        {"max", reduction(kael::max)},
        {"min", reduction(kael::min)},
        {"all", reduction(kael::all)},
        {"any", reduction(kael::any)},
        {"norm", reduction(kael::norm)},
        {"var", reduction_with_correction(kael::var, 1)},
        {"std", reduction_with_correction(kael::std, 1)},
        {"var_correction0", reduction_with_correction(kael::var, 0)},
        {"argmax", arg_reduction(kael::argmax, kael::argmax)},
        {"argmin", arg_reduction(kael::argmin, kael::argmin)},
        {"matmul", tensor_pair(kael::matmul)},
        {"dot", tensor_pair(kael::dot)},
        {"outer", tensor_pair(kael::outer)},
        {"cat",
         [](const Inputs& inputs, const Args& args) {
             return kael::cat(inputs, args.integer("dim"));
         }},
        {"stack",
         [](const Inputs& inputs, const Args& args) {
             return kael::stack(inputs, args.integer("dim"));
         }},
        {"where",
         [](const Inputs& inputs, const Args&) {
             return kael::where(inputs[0], inputs[1], inputs[2]);
         }},
        {"masked_fill",
         [](const Inputs& inputs, const Args& args) {
             return kael::masked_fill(inputs[0], inputs[1], args.real("value"));
         }},
        {"index_select",
         [](const Inputs& inputs, const Args& args) {
             return kael::index_select(inputs[0], args.integer("dim"), inputs[1]);
         }},
        {"gather",
         [](const Inputs& inputs, const Args& args) {
             return kael::gather(inputs[0], args.integer("dim"), inputs[1]);
         }},
        {"scatter_add_",
         [](const Inputs& inputs, const Args& args) {
             Tensor self = inputs[0];
             return self.scatter_add_(args.integer("dim"), inputs[1], inputs[2]);
         }},
        {"logsumexp",
         [](const Inputs& inputs, const Args& args) {
             return kael::logsumexp(inputs[0], args.integer("dim"), args.flag("keepdim"));
         }},
        {"softmax",
         [](const Inputs& inputs, const Args& args) {
             return kael::softmax(inputs[0], args.integer("dim"));
         }},
        {"log_softmax",
         [](const Inputs& inputs, const Args& args) {
             return kael::log_softmax(inputs[0], args.integer("dim"));
         }},
        {"clamp",
         [](const Inputs& inputs, const Args& args) {
             return kael::clamp(inputs[0], args.real("min"), args.real("max"));
         }},
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
