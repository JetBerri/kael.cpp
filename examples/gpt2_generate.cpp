#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/kael.h"
#include "kael/nn/transformer.h"
#include "kael/text/bpe.h"

namespace {

struct Options {
    std::string prompt = "The meaning of life is";
    std::filesystem::path directory = "data/gpt2";
    kael::Device device = kael::Device::CPU;
    std::int64_t tokens = 60;
    double temperature = 0.8;
    std::int64_t top_k = 40;
    std::uint64_t seed = 42;
    bool help = false;
};

constexpr const char* kUsage =
    "usage: gpt2_generate [PROMPT] [--data DIR] [--device cpu|cuda] [--tokens N]\n"
    "                     [--temperature T] [--top-k K] [--seed S]\n";

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const bool has_value = i + 1 < argc;
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--tokens" && has_value) {
            options.tokens = std::stoll(argv[++i]);
        } else if (argument == "--temperature" && has_value) {
            options.temperature = std::stod(argv[++i]);
        } else if (argument == "--top-k" && has_value) {
            options.top_k = std::stoll(argv[++i]);
        } else if (argument == "--seed" && has_value) {
            options.seed = std::stoull(argv[++i]);
        } else if (argument == "--data" && has_value) {
            options.directory = argv[++i];
        } else if (argument == "--device" && has_value) {
            options.device = kael::parse_device(argv[++i]);
        } else if (argument.starts_with("-")) {
            throw std::invalid_argument("unknown option or missing value: " + argument + "\n" +
                                        kUsage);
        } else {
            options.prompt = argument;
        }
    }
    return options;
}

void run(const Options& options) {
    const kael::text::BpeTokenizer tokenizer(options.directory / "vocab.json",
                                             options.directory / "merges.txt");
    kael::nn::GPT model(kael::nn::GptConfig::gpt2_small());
    kael::nn::load_gpt2_weights(model, options.directory / "model.safetensors");
    model.to(options.device);
    model.eval();

    const std::vector<std::int64_t> ids = tokenizer.encode(options.prompt);
    const auto length = static_cast<std::int64_t>(ids.size());
    const kael::Tensor prompt =
        kael::from_vector<std::int64_t>(ids, {1, length}).to(options.device);

    kael::manual_seed(options.seed);
    const auto started = std::chrono::steady_clock::now();
    const kael::Tensor output = kael::nn::generate(model, prompt,
                                                   {.max_new_tokens = options.tokens,
                                                    .temperature = options.temperature,
                                                    .top_k = options.top_k});
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;

    const kael::Tensor sequence = output.select(0, 0).cpu();
    const auto* values = sequence.data_ptr<std::int64_t>();
    std::cout << tokenizer.decode({values, values + sequence.numel()}) << '\n';
    std::cerr << static_cast<double>(options.tokens) / elapsed.count() << " tokens/s\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            std::cout << kUsage;
            return 0;
        }
        run(options);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
