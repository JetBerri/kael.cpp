#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "kael/nn/activation.h"
#include "kael/nn/dropout.h"
#include "kael/nn/embedding.h"
#include "kael/nn/layernorm.h"
#include "kael/nn/linear.h"
#include "kael/nn/module.h"

namespace kael::nn {

struct GptConfig {
    std::int64_t vocab_size = 50257;
    std::int64_t max_positions = 1024;
    std::int64_t layers = 12;
    std::int64_t heads = 12;
    std::int64_t dimension = 768;
    double dropout = 0.0;
    bool tanh_gelu = true;
    DType dtype = DType::F32;

    [[nodiscard]] static GptConfig gpt2_small() { return {}; }
};

struct LayerCache {
    std::optional<Tensor> keys;
    std::optional<Tensor> values;
};

using KvCache = std::vector<LayerCache>;

class CausalSelfAttention : public UnaryModule {
public:
    CausalSelfAttention(std::int64_t dimension, std::int64_t heads, std::int64_t max_positions,
                        double dropout = 0.0, DType dtype = DType::F32);

    Tensor forward(const Tensor& input) override { return forward(input, nullptr); }
    Tensor forward(const Tensor& input, LayerCache* cache);

    [[nodiscard]] Linear& projection() { return *projection_; }

private:
    std::int64_t dimension_;
    std::int64_t heads_;
    std::int64_t head_dimension_;
    std::int64_t max_positions_;
    double dropout_;
    Tensor future_mask_;
    std::shared_ptr<Linear> query_key_value_;
    std::shared_ptr<Linear> projection_;
};

class TransformerMlp : public UnaryModule {
public:
    explicit TransformerMlp(const GptConfig& config);
    Tensor forward(const Tensor& input) override;

    [[nodiscard]] Linear& output_layer() { return *fc2_; }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<GELU> activation_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Dropout> dropout_;
};

class TransformerBlock : public Module {
public:
    explicit TransformerBlock(const GptConfig& config);
    Tensor forward(const Tensor& input, LayerCache* cache = nullptr);

    [[nodiscard]] CausalSelfAttention& attention() { return *attention_; }
    [[nodiscard]] TransformerMlp& mlp() { return *mlp_; }

private:
    std::shared_ptr<LayerNorm> ln1_;
    std::shared_ptr<CausalSelfAttention> attention_;
    std::shared_ptr<LayerNorm> ln2_;
    std::shared_ptr<TransformerMlp> mlp_;
};

class GPT : public Module {
public:
    explicit GPT(const GptConfig& config);

    Tensor operator()(const Tensor& indices) { return forward(indices); }
    Tensor forward(const Tensor& indices, KvCache* cache = nullptr);

    [[nodiscard]] const GptConfig& config() const noexcept { return config_; }
    [[nodiscard]] const Tensor& token_embedding_weight() const {
        return token_embedding_->weight();
    }

private:
    void initialize_weights();

    GptConfig config_;
    std::shared_ptr<Embedding> token_embedding_;
    std::shared_ptr<Embedding> position_embedding_;
    std::shared_ptr<Dropout> dropout_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<LayerNorm> ln_final_;
};

struct GenerateOptions {
    std::int64_t max_new_tokens = 16;
    double temperature = 1.0;
    std::int64_t top_k = 0;
    bool use_cache = true;
};

[[nodiscard]] Tensor generate(GPT& model, const Tensor& prompt, const GenerateOptions& options);

void load_gpt2_weights(GPT& model, const std::filesystem::path& path);

} // namespace kael::nn
