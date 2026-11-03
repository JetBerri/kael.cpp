#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <stdexcept>

#include <gtest/gtest.h>

#include "kael/autograd/gradcheck.h"
#include "kael/io/safetensors.h"
#include "kael/kael.h"
#include "kael/nn/loss.h"
#include "kael/nn/transformer.h"
#include "kael/optim/adam.h"

#include "support/allclose.h"

using kael::DType;
using kael::Tensor;
using kael::nn::GPT;
using kael::nn::GptConfig;

namespace {

GptConfig tiny_config(DType dtype = DType::F64) {
    GptConfig config;
    config.vocab_size = 11;
    config.max_positions = 16;
    config.layers = 2;
    config.heads = 2;
    config.dimension = 8;
    config.dtype = dtype;
    return config;
}

Tensor tokens(const std::vector<std::int64_t>& values) {
    return kael::from_vector(values, {1, static_cast<std::int64_t>(values.size())});
}

} // namespace

TEST(Attention, OutputKeepsTheInputShape) {
    kael::manual_seed(1);
    kael::nn::CausalSelfAttention attention(8, 2, 16, 0.0, DType::F64);
    EXPECT_EQ(attention(kael::randn({3, 5, 8}, DType::F64)).shape(), (kael::Shape{3, 5, 8}));
    EXPECT_THROW((void)attention(kael::randn({3, 17, 8}, DType::F64)), kael::ValueError);
    EXPECT_THROW((void)attention(kael::randn({3, 8}, DType::F64)), kael::ValueError);
}

TEST(Attention, DimensionMustDivideByHeads) {
    EXPECT_THROW(kael::nn::CausalSelfAttention(7, 2, 16), kael::ValueError);
    EXPECT_THROW(kael::nn::CausalSelfAttention(8, 0, 16), kael::ValueError);
}

TEST(Attention, LaterPositionsDoNotAffectEarlierOnes) {
    kael::manual_seed(2);
    kael::nn::CausalSelfAttention attention(8, 2, 16, 0.0, DType::F64);
    const Tensor x = kael::randn({1, 6, 8}, DType::F64);
    Tensor changed = x.clone();
    changed.select(1, 5).add_(100.0);
    const Tensor original_output = attention(x);
    const Tensor changed_output = attention(changed);
    EXPECT_TRUE(kael::test::allclose(original_output.narrow(1, 0, 5).detach(),
                                     changed_output.narrow(1, 0, 5).detach()));
    EXPECT_FALSE(
        kael::equal(original_output.select(1, 5).detach(), changed_output.select(1, 5).detach()));
}

TEST(Attention, Gradcheck) {
    kael::manual_seed(3);
    kael::nn::CausalSelfAttention attention(4, 2, 8, 0.0, DType::F64);
    Tensor x = kael::randn({1, 3, 4}, DType::F64);
    x.requires_grad_();
    const auto result = kael::autograd::gradcheck(
        [&](const std::vector<Tensor>& in) { return attention(in[0]); }, {x});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(Block, GradcheckThroughTheWholeBlock) {
    kael::manual_seed(4);
    kael::nn::TransformerBlock block(tiny_config());
    Tensor x = kael::randn({1, 2, 8}, DType::F64);
    x.requires_grad_();
    const auto result = kael::autograd::gradcheck(
        [&](const std::vector<Tensor>& in) { return block.forward(in[0]); }, {x});
    EXPECT_TRUE(result.passed) << result.report;
}

TEST(Gpt, LogitsHaveTheExpectedShape) {
    kael::manual_seed(5);
    GPT model(tiny_config());
    EXPECT_EQ(model(kael::randint(0, 11, {2, 7})).shape(), (kael::Shape{2, 7, 11}));
}

TEST(Gpt, RejectsSequencesThatDoNotFit) {
    GPT model(tiny_config());
    EXPECT_THROW((void)model(kael::zeros({1, 17}, DType::I64)), kael::ValueError);
    EXPECT_THROW((void)model(kael::zeros({17}, DType::I64)), kael::ValueError);
    EXPECT_THROW((void)model(kael::zeros({1, 2})), kael::TypeError);
    EXPECT_THROW((void)model(kael::full({1, 2}, 11, DType::I64)), kael::IndexError);
}

TEST(Gpt, ParameterNamesFollowTheGuide) {
    GPT model(tiny_config());
    const auto state = model.state_dict();
    for (const char* name :
         {"token_embedding.weight", "position_embedding.weight", "blocks.0.ln1.weight",
          "blocks.0.ln1.bias", "blocks.1.attn.qkv.weight", "blocks.1.attn.proj.bias",
          "blocks.0.ln2.weight", "blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc2.bias",
          "ln_final.weight", "ln_final.bias"}) {
        EXPECT_TRUE(state.contains(name)) << name;
    }
}

TEST(Gpt, WeightTyingKeepsASingleEmbeddingParameter) {
    GPT model(tiny_config());
    const auto named = model.named_parameters();
    EXPECT_EQ(std::count_if(named.begin(), named.end(),
                            [&](const auto& entry) {
                                return entry.second.shares_storage(model.token_embedding_weight());
                            }),
              1);
}

TEST(Gpt, ParameterCountFormula) {
    const GptConfig config = tiny_config();
    GPT model(config);
    const std::int64_t c = config.dimension;
    const std::int64_t per_block =
        (3 * c * c + 3 * c) + (c * c + c) + (4 * c * c + 4 * c) + (4 * c * c + c) + 2 * (2 * c);
    const std::int64_t expected =
        config.vocab_size * c + config.max_positions * c + config.layers * per_block + 2 * c;
    EXPECT_EQ(model.num_parameters(), expected);
}

TEST(Gpt, Gpt2SmallHasTheKnownParameterCount) {
    if (!kael::caching_allocator_enabled()) {
        GTEST_SKIP() << "drawing 250 million samples is too slow under sanitizers";
    }
    GPT model(GptConfig::gpt2_small());
    EXPECT_EQ(model.num_parameters(), 124439808);
}

TEST(Gpt, InitialLossMatchesUniformPrediction) {
    kael::manual_seed(6);
    GptConfig config = tiny_config();
    config.vocab_size = 65;
    GPT model(config);
    const Tensor input = kael::randint(0, 65, {8, 12});
    const Tensor logits = model(input);
    const Tensor loss = kael::nn::cross_entropy(logits.view({96, 65}), kael::randint(0, 65, {96}));
    EXPECT_NEAR(loss.item<double>(), std::log(65.0), 0.1);
}

TEST(Gpt, ResidualProjectionsStartSmaller) {
    kael::manual_seed(7);
    GptConfig config = tiny_config();
    config.layers = 8;
    config.dimension = 64;
    config.heads = 4;
    GPT model(config);
    const auto state = model.state_dict();
    const double projection = kael::std(state.at("blocks.0.attn.proj.weight")).item<double>();
    const double regular = kael::std(state.at("blocks.0.attn.qkv.weight")).item<double>();
    EXPECT_NEAR(regular, 0.02, 0.002);
    EXPECT_NEAR(projection, 0.02 / std::sqrt(16.0), 0.001);
    EXPECT_EQ(kael::max(kael::abs(state.at("blocks.0.attn.qkv.bias"))).item<double>(), 0.0);
}

TEST(Gpt, OverfitsASmallSequence) {
    kael::manual_seed(8);
    GPT model(tiny_config(DType::F32));
    kael::optim::AdamWOptions options;
    options.learning_rate = 3e-3;
    kael::optim::AdamW optimizer(model.parameters(), options);
    const Tensor sequence = tokens({1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Tensor input = sequence.narrow(1, 0, 8);
    const Tensor target = sequence.narrow(1, 1, 8).reshape({8});
    double loss_value = 0.0;
    for (int step = 0; step < 300; ++step) {
        const Tensor loss = kael::nn::cross_entropy(model(input).view({8, 11}), target);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        loss_value = loss.item();
    }
    EXPECT_LT(loss_value, 0.05);
}

TEST(Gpt, CacheGivesTheSameLogitsAsTheFullSequence) {
    kael::manual_seed(9);
    GPT model(tiny_config());
    model.eval();
    const kael::autograd::NoGradGuard guard;
    const Tensor sequence = kael::randint(0, 11, {2, 9});
    const Tensor full = model(sequence);

    kael::nn::KvCache cache;
    const Tensor prefix = model.forward(sequence.narrow(1, 0, 4), &cache);
    EXPECT_TRUE(kael::test::allclose(prefix, full.narrow(1, 0, 4)));
    for (std::int64_t position = 4; position < 9; ++position) {
        const Tensor step = model.forward(sequence.narrow(1, position, 1), &cache);
        EXPECT_TRUE(kael::test::allclose(step.select(1, 0), full.select(1, position)))
            << "position " << position;
    }
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_EQ(cache[0].keys->size(2), 9);
}

TEST(Gpt, CacheCannotOutgrowTheContext) {
    GPT model(tiny_config());
    kael::nn::KvCache cache;
    const kael::autograd::NoGradGuard guard;
    (void)model.forward(kael::zeros({1, 16}, DType::I64), &cache);
    EXPECT_THROW((void)model.forward(kael::zeros({1, 1}, DType::I64), &cache), kael::ValueError);
}

TEST(Generate, GreedyMatchesWithAndWithoutCache) {
    kael::manual_seed(10);
    GPT model(tiny_config());
    const Tensor prompt = tokens({1, 2, 3});
    const auto generate = [&](bool use_cache) {
        return kael::nn::generate(
            model, prompt, {.max_new_tokens = 6, .temperature = 0.0, .use_cache = use_cache});
    };
    const Tensor cached = generate(true);
    EXPECT_EQ(cached.shape(), (kael::Shape{1, 9}));
    EXPECT_TRUE(kael::equal(cached, generate(false)));
    EXPECT_TRUE(kael::equal(cached.narrow(1, 0, 3), prompt));
}

TEST(Generate, SamplingMatchesWithAndWithoutCacheForTheSameSeed) {
    GPT model(tiny_config());
    const Tensor prompt = tokens({1, 2, 3});
    const auto sample = [&](bool use_cache) {
        kael::manual_seed(11);
        return kael::nn::generate(
            model, prompt,
            {.max_new_tokens = 6, .temperature = 0.8, .top_k = 5, .use_cache = use_cache});
    };
    EXPECT_TRUE(kael::equal(sample(true), sample(false)));
}

TEST(Generate, KeepsGoingPastTheContextLength) {
    kael::manual_seed(12);
    GPT model(tiny_config());
    for (const bool use_cache : {true, false}) {
        const Tensor out =
            kael::nn::generate(model, tokens({1, 2}),
                               {.max_new_tokens = 40, .temperature = 1.0, .use_cache = use_cache});
        EXPECT_EQ(out.size(1), 42);
    }
}

TEST(Generate, TopKOneIsGreedy) {
    GPT model(tiny_config());
    const Tensor prompt = tokens({4, 5});
    kael::manual_seed(13);
    const Tensor sampled =
        kael::nn::generate(model, prompt, {.max_new_tokens = 5, .temperature = 1.0, .top_k = 1});
    const Tensor greedy =
        kael::nn::generate(model, prompt, {.max_new_tokens = 5, .temperature = 0.0});
    EXPECT_TRUE(kael::equal(sampled, greedy));
}

TEST(Generate, RestoresTrainingModeAndHandlesBatches) {
    kael::manual_seed(14);
    GPT model(tiny_config());
    model.train();
    const Tensor out = kael::nn::generate(model, kael::zeros({3, 2}, DType::I64),
                                          {.max_new_tokens = 2, .temperature = 1.0});
    EXPECT_EQ(out.shape(), (kael::Shape{3, 4}));
    EXPECT_TRUE(model.is_training());
}

TEST(Generate, ValidatesOptions) {
    GPT model(tiny_config());
    EXPECT_THROW((void)kael::nn::generate(model, tokens({1}), {.max_new_tokens = -1}),
                 kael::ValueError);
    EXPECT_THROW((void)kael::nn::generate(model, tokens({1}), {.temperature = -1.0}),
                 kael::ValueError);
    EXPECT_THROW((void)kael::nn::generate(model, tokens({1}), {.top_k = -1}), kael::ValueError);
}

TEST(Gpt2Weights, MissingEntriesAreReported) {
    const auto path = std::filesystem::temp_directory_path() / "kael_gpt2_missing.safetensors";
    kael::io::save_safetensors(path, {{"wte.weight", kael::zeros({11, 8})}});
    GPT model(tiny_config(DType::F32));
    EXPECT_THROW(kael::nn::load_gpt2_weights(model, path), kael::Error);
    std::filesystem::remove(path);
}

TEST(Gpt2Weights, Conv1DWeightsAreTransposed) {
    GPT source(tiny_config(DType::F32));
    std::map<std::string, Tensor> hugging_face;
    const auto state = source.state_dict();
    hugging_face.emplace("wte.weight", state.at("token_embedding.weight"));
    hugging_face.emplace("wpe.weight", state.at("position_embedding.weight"));
    hugging_face.emplace("ln_f.weight", state.at("ln_final.weight"));
    hugging_face.emplace("ln_f.bias", state.at("ln_final.bias"));
    hugging_face.emplace("lm_head.weight", state.at("token_embedding.weight"));
    for (const char* index : {"0", "1"}) {
        const std::string hf = std::string("transformer.h.") + index + ".";
        const std::string own = std::string("blocks.") + index + ".";
        hugging_face.emplace(hf + "ln_1.weight", state.at(own + "ln1.weight"));
        hugging_face.emplace(hf + "ln_1.bias", state.at(own + "ln1.bias"));
        hugging_face.emplace(hf + "ln_2.weight", state.at(own + "ln2.weight"));
        hugging_face.emplace(hf + "ln_2.bias", state.at(own + "ln2.bias"));
        hugging_face.emplace(hf + "attn.c_attn.weight", state.at(own + "attn.qkv.weight").T());
        hugging_face.emplace(hf + "attn.c_attn.bias", state.at(own + "attn.qkv.bias"));
        hugging_face.emplace(hf + "attn.c_proj.weight", state.at(own + "attn.proj.weight").T());
        hugging_face.emplace(hf + "attn.c_proj.bias", state.at(own + "attn.proj.bias"));
        hugging_face.emplace(hf + "mlp.c_fc.weight", state.at(own + "mlp.fc1.weight").T());
        hugging_face.emplace(hf + "mlp.c_fc.bias", state.at(own + "mlp.fc1.bias"));
        hugging_face.emplace(hf + "mlp.c_proj.weight", state.at(own + "mlp.fc2.weight").T());
        hugging_face.emplace(hf + "mlp.c_proj.bias", state.at(own + "mlp.fc2.bias"));
        hugging_face.emplace(hf + "attn.bias", kael::zeros({1}));
    }
    const auto path = std::filesystem::temp_directory_path() / "kael_gpt2_tiny.safetensors";
    kael::io::save_safetensors(path, hugging_face);
    GPT target(tiny_config(DType::F32));
    kael::nn::load_gpt2_weights(target, path);
    std::filesystem::remove(path);
    const auto loaded = target.state_dict();
    for (const auto& [name, tensor] : state) {
        EXPECT_TRUE(kael::equal(loaded.at(name), tensor)) << name;
    }
}

TEST(Gpt, NeedsAtLeastOneLayer) {
    GptConfig config = tiny_config();
    config.layers = 0;
    EXPECT_THROW(GPT model(config), kael::ValueError);
}

TEST(Gpt, RejectsACacheOfAnotherModel) {
    GPT model(tiny_config());
    kael::nn::KvCache cache(1);
    const kael::autograd::NoGradGuard guard;
    EXPECT_THROW((void)model.forward(kael::zeros({1, 2}, DType::I64), &cache), kael::ValueError);
}

TEST(GptGenerate, ReportsEveryNewTokenInOrder) {
    kael::manual_seed(61);
    GPT model(tiny_config());
    const kael::Tensor prompt = kael::zeros({1, 2}, kael::DType::I64);
    std::vector<std::int64_t> seen;
    const kael::Tensor output =
        kael::nn::generate(model, prompt,
                           {.max_new_tokens = 5,
                            .on_token = [&](const kael::Tensor& token) {
                                seen.push_back(token.item<std::int64_t>());
                                return true;
                            }});
    ASSERT_EQ(seen.size(), 5u);
    for (std::size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(output.at<std::int64_t>({0, static_cast<std::int64_t>(i) + 2}), seen[i]);
    }
}

TEST(GptGenerate, AReceiverCanStopTheGeneration) {
    kael::manual_seed(62);
    GPT model(tiny_config());
    const kael::Tensor prompt = kael::zeros({1, 2}, kael::DType::I64);
    int calls = 0;
    const kael::Tensor output = kael::nn::generate(model, prompt,
                                                   {.max_new_tokens = 50,
                                                    .on_token = [&](const kael::Tensor&) {
                                                        ++calls;
                                                        return calls < 3;
                                                    }});
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(output.size(1), 5);
}

TEST(GptGenerate, AThrowingReceiverRestoresTheTrainingMode) {
    kael::manual_seed(63);
    GPT model(tiny_config());
    model.train();
    const kael::Tensor prompt = kael::zeros({1, 2}, kael::DType::I64);
    EXPECT_THROW((void)kael::nn::generate(model, prompt,
                                          {.max_new_tokens = 2,
                                           .on_token = [](const kael::Tensor&) -> bool {
                                               throw std::runtime_error("stop");
                                           }}),
                 std::runtime_error);
    EXPECT_TRUE(model.is_training());
}
