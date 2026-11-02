#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "kael/io/safetensors.h"
#include "kael/kael.h"
#include "kael/nn/loss.h"
#include "kael/nn/transformer.h"
#include "kael/optim/adam.h"
#include "kael/optim/clip.h"
#include "kael/optim/scheduler.h"

using kael::Tensor;

namespace {

constexpr std::int64_t kBatchSize = 32;
constexpr std::int64_t kContext = 64;
constexpr std::int64_t kWarmupSteps = 200;
constexpr std::int64_t kEvaluateEvery = 250;
constexpr std::int64_t kEvaluationBatches = 20;
constexpr std::int64_t kSampleLength = 500;
constexpr double kLearningRate = 1e-3;
constexpr double kMaxGradientNorm = 1.0;

struct Options {
    std::filesystem::path directory = "data/shakespeare";
    std::int64_t steps = 5000;
    kael::Device device = kael::Device::CPU;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--steps" && i + 1 < argc) {
            options.steps = std::stoll(argv[++i]);
        } else if (argument == "--device" && i + 1 < argc) {
            options.device =
                std::string(argv[++i]) == "cuda" ? kael::Device::CUDA : kael::Device::CPU;
        } else {
            options.directory = argument;
        }
    }
    return options;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    KAEL_CHECK(file.is_open(), "cannot open " << path << "; run scripts/download_shakespeare.sh");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class Vocabulary {
public:
    explicit Vocabulary(const std::string& text) {
        std::array<bool, 256> seen{};
        for (const char c : text) {
            seen[static_cast<unsigned char>(c)] = true;
        }
        for (std::size_t code = 0; code < seen.size(); ++code) {
            if (seen[code]) {
                codes_[static_cast<char>(code)] = static_cast<std::int64_t>(characters_.size());
                characters_.push_back(static_cast<char>(code));
            }
        }
    }

    [[nodiscard]] std::int64_t size() const {
        return static_cast<std::int64_t>(characters_.size());
    }

    [[nodiscard]] std::vector<std::int64_t> encode(const std::string& text) const {
        std::vector<std::int64_t> encoded;
        encoded.reserve(text.size());
        for (const char c : text) {
            encoded.push_back(codes_.at(c));
        }
        return encoded;
    }

    [[nodiscard]] std::string decode(const Tensor& tokens) const {
        std::string text;
        for (std::int64_t i = 0; i < tokens.numel(); ++i) {
            text.push_back(characters_[static_cast<std::size_t>(tokens.at<std::int64_t>({i}))]);
        }
        return text;
    }

private:
    std::vector<char> characters_;
    std::map<char, std::int64_t> codes_;
};

struct Batch {
    Tensor inputs;
    Tensor targets;
};

Batch sample_batch(const std::vector<std::int64_t>& data, kael::Device device) {
    const Tensor starts =
        kael::randint(0, static_cast<std::int64_t>(data.size()) - kContext - 1, {kBatchSize});
    Tensor inputs = kael::empty({kBatchSize, kContext}, kael::DType::I64);
    Tensor targets = kael::empty({kBatchSize * kContext}, kael::DType::I64);
    auto* input_values = inputs.data_ptr<std::int64_t>();
    auto* target_values = targets.data_ptr<std::int64_t>();
    for (std::int64_t row = 0; row < kBatchSize; ++row) {
        const auto start = data.begin() + starts.at<std::int64_t>({row});
        std::copy_n(start, kContext, input_values + row * kContext);
        std::copy_n(start + 1, kContext, target_values + row * kContext);
    }
    return {inputs.to(device), targets.to(device)};
}

Tensor batch_loss(kael::nn::GPT& model, const Batch& batch, std::int64_t vocabulary) {
    return kael::nn::cross_entropy(model(batch.inputs).view({kBatchSize * kContext, vocabulary}),
                                   batch.targets);
}

double evaluate(kael::nn::GPT& model, const std::vector<std::int64_t>& data,
                std::int64_t vocabulary, kael::Device device) {
    const kael::autograd::NoGradGuard guard;
    model.eval();
    double total = 0.0;
    for (std::int64_t i = 0; i < kEvaluationBatches; ++i) {
        total += batch_loss(model, sample_batch(data, device), vocabulary).item();
    }
    model.train();
    return total / static_cast<double>(kEvaluationBatches);
}

void run(const Options& options) {
    kael::manual_seed(1337);
    const std::string text = read_text(options.directory / "input.txt");
    const Vocabulary vocabulary(text);
    const std::vector<std::int64_t> encoded = vocabulary.encode(text);
    const auto split = static_cast<std::ptrdiff_t>(encoded.size() * 9 / 10);
    const std::vector<std::int64_t> train(encoded.begin(), encoded.begin() + split);
    const std::vector<std::int64_t> validation(encoded.begin() + split, encoded.end());

    kael::nn::GptConfig config;
    config.vocab_size = vocabulary.size();
    config.max_positions = kContext;
    config.layers = 4;
    config.heads = 4;
    config.dimension = 128;
    config.tanh_gelu = false;
    kael::nn::GPT model(config);
    model.to(options.device);
    std::cout << "vocabulary " << vocabulary.size() << ", parameters " << model.num_parameters()
              << "\n";

    kael::optim::AdamWOptions adam;
    adam.learning_rate = kLearningRate;
    kael::optim::AdamW optimizer(model.parameters(), adam);
    kael::optim::WarmupCosineLR schedule(optimizer, kWarmupSteps, options.steps);

    const std::filesystem::path best_path = options.directory / "best.safetensors";
    double best_loss = std::numeric_limits<double>::infinity();
    const auto started = std::chrono::steady_clock::now();
    for (std::int64_t step = 0; step <= options.steps; ++step) {
        if (step % kEvaluateEvery == 0) {
            const double validation_loss =
                evaluate(model, validation, vocabulary.size(), options.device);
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - started;
            std::cout << "step " << step << " validation loss " << validation_loss << " after "
                      << elapsed.count() << " s" << std::endl;
            if (validation_loss < best_loss) {
                best_loss = validation_loss;
                kael::io::save_safetensors(best_path, model.state_dict());
            }
        }
        if (step == options.steps) {
            break;
        }
        const Tensor loss =
            batch_loss(model, sample_batch(train, options.device), vocabulary.size());
        optimizer.zero_grad();
        loss.backward();
        kael::optim::clip_grad_norm_(model.parameters(), kMaxGradientNorm);
        optimizer.step();
        schedule.step();
    }

    model.load_state_dict(kael::io::load_safetensors(best_path));
    const Tensor prompt =
        kael::from_vector(vocabulary.encode("ROMEO:\n"), {1, 7}).to(options.device);
    const Tensor generated = kael::nn::generate(
        model, prompt, {.max_new_tokens = kSampleLength, .temperature = 0.8, .top_k = 40});
    std::cout << "best validation loss " << best_loss << "\n"
              << vocabulary.decode(generated.select(0, 0).cpu()) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
