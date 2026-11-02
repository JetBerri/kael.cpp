#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "kael/data/dataloader.h"
#include "kael/data/mnist.h"
#include "kael/kael.h"
#include "kael/nn/activation.h"
#include "kael/nn/linear.h"
#include "kael/nn/loss.h"
#include "kael/nn/sequential.h"
#include "kael/optim/adam.h"

using kael::Tensor;
using kael::nn::Linear;

namespace {

constexpr std::uint64_t kSeed = 42;
constexpr int kEpochs = 5;
constexpr std::int64_t kBatchSize = 64;
constexpr std::int64_t kReportEvery = 100;

double accuracy(kael::nn::Sequential& model, const kael::data::Dataset& dataset) {
    model.eval();
    const kael::autograd::NoGradGuard guard;
    std::int64_t correct = 0;
    for (const auto& [images, labels] : kael::data::DataLoader(dataset, {.batch_size = 1000})) {
        correct += kael::sum(kael::argmax(model(images), 1) == labels).item<std::int64_t>();
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

void train_epoch(kael::nn::Sequential& model, kael::optim::Optimizer& optimizer,
                 const kael::data::DataLoader& loader, int epoch) {
    model.train();
    std::int64_t step = 0;
    for (const auto& [images, labels] : loader) {
        const Tensor loss = kael::nn::cross_entropy(model(images), labels);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        if (step % kReportEvery == 0) {
            std::cout << "epoch " << epoch << " step " << step << " loss " << loss.item() << '\n';
        }
        ++step;
    }
}

void run(const std::filesystem::path& directory) {
    kael::manual_seed(kSeed);
    const auto train = kael::data::load_mnist(directory, kael::data::MnistPart::Train);
    const auto test = kael::data::load_mnist(directory, kael::data::MnistPart::Test);
    const kael::data::TensorDataset train_set(train.images, train.labels);
    const kael::data::TensorDataset test_set(test.images, test.labels);

    kael::nn::Sequential model{
        std::make_shared<kael::nn::Flatten>(), std::make_shared<Linear>(784, 128),
        std::make_shared<kael::nn::ReLU>(), std::make_shared<Linear>(128, 10)};
    kael::optim::AdamW optimizer(model.parameters());
    const kael::data::DataLoader loader(train_set, {.batch_size = kBatchSize, .shuffle = true});

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        const auto start = std::chrono::steady_clock::now();
        train_epoch(model, optimizer, loader, epoch);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::cout << "epoch " << epoch << " test accuracy " << accuracy(model, test_set) * 100.0
                  << "% in " << elapsed.count() << " s\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        run(argc > 1 ? argv[1] : "data/mnist");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
