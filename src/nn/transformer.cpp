#include "kael/nn/transformer.h"

#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "kael/autograd/grad_mode.h"
#include "kael/factory.h"
#include "kael/io/safetensors.h"
#include "kael/nn/functional.h"
#include "kael/nn/init.h"
#include "kael/ops.h"
#include "kael/random.h"

namespace kael::nn {

namespace {

constexpr double kInitialStd = 0.02;

Tensor future_positions_mask(std::int64_t size) {
    return triu(ones({size, size}, DType::Bool), 1);
}

const GptConfig& checked_config(const GptConfig& config) {
    KAEL_CHECK_VALUE(config.layers > 0, "GPT needs at least one layer, got " << config.layers);
    return config;
}

std::int64_t checked_head_dimension(std::int64_t dimension, std::int64_t heads) {
    KAEL_CHECK_VALUE(heads > 0 && dimension % heads == 0,
                     "dimension " << dimension << " does not divide by " << heads << " heads");
    return dimension / heads;
}

Tensor sample_next(const Tensor& last_logits, const GenerateOptions& options) {
    if (options.temperature == 0.0) {
        return argmax(last_logits, -1, true);
    }
    Tensor logits = last_logits / options.temperature;
    if (options.top_k > 0 && options.top_k < logits.size(-1)) {
        const Tensor smallest_kept =
            topk(logits, options.top_k).values.select(-1, options.top_k - 1);
        logits = masked_fill(logits, logits < smallest_kept.unsqueeze(-1),
                             -std::numeric_limits<double>::infinity());
    }
    return multinomial(softmax(logits, -1), 1, true);
}

std::string translated_gpt2_name(std::string name) {
    const std::string prefix = "transformer.";
    if (name.rfind(prefix, 0) == 0) {
        name.erase(0, prefix.size());
    }
    const auto replace = [&](const std::string& from, const std::string& to) {
        const auto position = name.find(from);
        if (position != std::string::npos) {
            name.replace(position, from.size(), to);
        }
    };
    if (name.rfind("wte.", 0) == 0) {
        replace("wte.", "token_embedding.");
    } else if (name.rfind("wpe.", 0) == 0) {
        replace("wpe.", "position_embedding.");
    } else if (name.rfind("ln_f.", 0) == 0) {
        replace("ln_f.", "ln_final.");
    } else if (name.rfind("h.", 0) == 0) {
        name.replace(0, 2, "blocks.");
        replace(".ln_1.", ".ln1.");
        replace(".ln_2.", ".ln2.");
        replace(".attn.c_attn.", ".attn.qkv.");
        replace(".attn.c_proj.", ".attn.proj.");
        replace(".mlp.c_fc.", ".mlp.fc1.");
        replace(".mlp.c_proj.", ".mlp.fc2.");
    }
    return name;
}

bool stored_transposed(const std::string& name) {
    const bool conv1d = name.find(".attn.qkv.") != std::string::npos ||
                        name.find(".attn.proj.") != std::string::npos ||
                        name.find(".mlp.fc1.") != std::string::npos ||
                        name.find(".mlp.fc2.") != std::string::npos;
    return conv1d && name.ends_with(".weight");
}

} // namespace

CausalSelfAttention::CausalSelfAttention(std::int64_t dimension, std::int64_t heads,
                                         std::int64_t max_positions, double dropout, DType dtype)
    : dimension_(dimension), heads_(heads),
      head_dimension_(checked_head_dimension(dimension, heads)), max_positions_(max_positions),
      dropout_(dropout), future_mask_(future_positions_mask(max_positions)),
      query_key_value_(
          register_module("qkv", std::make_shared<Linear>(dimension, 3 * dimension, true, dtype))),
      projection_(
          register_module("proj", std::make_shared<Linear>(dimension, dimension, true, dtype))) {}

Tensor CausalSelfAttention::forward(const Tensor& input, LayerCache* cache) {
    KAEL_CHECK_VALUE(input.dim() == 3 && input.size(2) == dimension_,
                     "attention needs an input of shape [B, T, " << dimension_ << "], got "
                                                                 << to_string(input.shape()));
    const std::int64_t batch = input.size(0);
    const std::int64_t length = input.size(1);
    const std::int64_t past = cache != nullptr && cache->keys ? cache->keys->size(2) : 0;
    KAEL_CHECK_VALUE(past + length <= max_positions_,
                     "sequence of " << past + length << " positions does not fit a context of "
                                    << max_positions_);

    const Tensor parts =
        query_key_value_->forward(input).view({batch, length, 3, heads_, head_dimension_});
    const Tensor queries = parts.select(2, 0).transpose(1, 2);
    Tensor keys = parts.select(2, 1).transpose(1, 2);
    Tensor values = parts.select(2, 2).transpose(1, 2);
    if (cache != nullptr) {
        if (cache->keys) {
            keys = cat({*cache->keys, keys}, 2);
            values = cat({*cache->values, values}, 2);
        }
        cache->keys = keys;
        cache->values = values;
    }

    if (future_mask_.device() != input.device()) {
        future_mask_ = future_mask_.to(input.device());
    }
    const Tensor mask = future_mask_.slice(0, past, past + length).slice(1, 0, past + length);
    const Tensor scores =
        matmul(queries, keys.mT()) / std::sqrt(static_cast<double>(head_dimension_));
    const Tensor masked = masked_fill(scores, mask, -std::numeric_limits<double>::infinity());
    const Tensor weights = functional::dropout(softmax(masked, -1), dropout_, is_training());
    const Tensor mixed = matmul(weights, values).transpose(1, 2).contiguous();
    const Tensor output = projection_->forward(mixed.view({batch, length, dimension_}));
    return functional::dropout(output, dropout_, is_training());
}

TransformerMlp::TransformerMlp(const GptConfig& config)
    : fc1_(register_module("fc1", std::make_shared<Linear>(config.dimension, 4 * config.dimension,
                                                           true, config.dtype))),
      activation_(register_module("activation", std::make_shared<GELU>(config.tanh_gelu))),
      fc2_(register_module("fc2", std::make_shared<Linear>(4 * config.dimension, config.dimension,
                                                           true, config.dtype))),
      dropout_(register_module("dropout", std::make_shared<Dropout>(config.dropout))) {}

Tensor TransformerMlp::forward(const Tensor& input) {
    return dropout_->forward(fc2_->forward(activation_->forward(fc1_->forward(input))));
}

TransformerBlock::TransformerBlock(const GptConfig& config)
    : ln1_(register_module(
          "ln1", std::make_shared<LayerNorm>(Shape{config.dimension}, 1e-5, config.dtype))),
      attention_(register_module("attn", std::make_shared<CausalSelfAttention>(
                                             config.dimension, config.heads, config.max_positions,
                                             config.dropout, config.dtype))),
      ln2_(register_module(
          "ln2", std::make_shared<LayerNorm>(Shape{config.dimension}, 1e-5, config.dtype))),
      mlp_(register_module("mlp", std::make_shared<TransformerMlp>(config))) {}

Tensor TransformerBlock::forward(const Tensor& input, LayerCache* cache) {
    const Tensor attended = input + attention_->forward(ln1_->forward(input), cache);
    return attended + mlp_->forward(ln2_->forward(attended));
}

GPT::GPT(const GptConfig& config)
    : config_(checked_config(config)),
      token_embedding_(register_module(
          "token_embedding",
          std::make_shared<Embedding>(config.vocab_size, config.dimension, config.dtype))),
      position_embedding_(register_module(
          "position_embedding",
          std::make_shared<Embedding>(config.max_positions, config.dimension, config.dtype))),
      dropout_(register_module("dropout", std::make_shared<Dropout>(config.dropout))),
      ln_final_(nullptr) {
    for (std::int64_t i = 0; i < config.layers; ++i) {
        blocks_.push_back(register_module("blocks." + std::to_string(i),
                                          std::make_shared<TransformerBlock>(config)));
    }
    ln_final_ = register_module(
        "ln_final", std::make_shared<LayerNorm>(Shape{config.dimension}, 1e-5, config.dtype));
    initialize_weights();
}

void GPT::initialize_weights() {
    const double residual_std =
        kInitialStd /
        std::sqrt(2.0 * static_cast<double>(std::max<std::int64_t>(config_.layers, 1)));
    Tensor token_weight = token_embedding_->weight();
    Tensor position_weight = position_embedding_->weight();
    init::normal_(token_weight, 0.0, kInitialStd);
    init::normal_(position_weight, 0.0, kInitialStd);
    const auto named = named_parameters();
    for (const auto& [name, parameter] : named) {
        if (name.rfind("blocks.", 0) != 0 || name.find(".ln") != std::string::npos) {
            continue;
        }
        Tensor target = parameter;
        if (name.ends_with(".bias")) {
            init::zeros_(target);
        } else {
            const bool residual =
                name.ends_with(".attn.proj.weight") || name.ends_with(".mlp.fc2.weight");
            init::normal_(target, 0.0, residual ? residual_std : kInitialStd);
        }
    }
}

Tensor GPT::forward(const Tensor& indices, KvCache* cache) {
    KAEL_CHECK_VALUE(indices.dim() == 2,
                     "GPT needs indices of shape [B, T], got " << to_string(indices.shape()));
    KAEL_CHECK_TYPE(indices.dtype() == DType::I64,
                    "GPT needs int64 indices, got " << indices.dtype());
    if (cache != nullptr && cache->empty()) {
        cache->resize(blocks_.size());
    }
    KAEL_CHECK_VALUE(cache == nullptr || cache->size() == blocks_.size(),
                     "cache has " << cache->size() << " layers but the model has "
                                  << blocks_.size());
    const std::int64_t past =
        cache != nullptr && cache->front().keys ? cache->front().keys->size(2) : 0;
    const std::int64_t length = indices.size(1);
    KAEL_CHECK_VALUE(past + length <= config_.max_positions,
                     "sequence of " << past + length << " positions does not fit a context of "
                                    << config_.max_positions);

    const Tensor positions =
        arange(static_cast<double>(past), static_cast<double>(past + length), 1.0, DType::I64)
            .to(indices.device());
    Tensor x = dropout_->forward(token_embedding_->forward(indices) +
                                 position_embedding_->forward(positions));
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        x = blocks_[i]->forward(x, cache != nullptr ? &(*cache)[i] : nullptr);
    }
    // Multiplying by the weight transposed would copy the whole embedding on every step.
    return matmul(token_embedding_->weight(), ln_final_->forward(x).mT()).mT().contiguous();
}

Tensor generate(GPT& model, const Tensor& prompt, const GenerateOptions& options) {
    KAEL_CHECK_VALUE(options.max_new_tokens >= 0,
                     "max_new_tokens must not be negative, got " << options.max_new_tokens);
    KAEL_CHECK_VALUE(options.temperature >= 0.0,
                     "temperature must not be negative, got " << options.temperature);
    KAEL_CHECK_VALUE(options.top_k >= 0, "top_k must not be negative, got " << options.top_k);
    KAEL_CHECK_VALUE(prompt.dim() == 2 && prompt.size(1) > 0,
                     "generate needs a prompt of shape [B, T], got " << to_string(prompt.shape()));

    const autograd::NoGradGuard guard;
    const bool was_training = model.is_training();
    model.eval();
    const std::int64_t context = model.config().max_positions;

    Tensor sequence = prompt;
    KvCache cache;
    for (std::int64_t step = 0; step < options.max_new_tokens; ++step) {
        const std::int64_t total = sequence.size(1);
        const std::int64_t window = std::min(total, context);
        const std::int64_t cached =
            !cache.empty() && cache.front().keys ? cache.front().keys->size(2) : 0;
        const auto run_model = [&] {
            if (!options.use_cache) {
                return model(sequence.narrow(1, total - window, window));
            }
            if (cached > 0 && cached < context) {
                return model.forward(sequence.narrow(1, total - 1, 1), &cache);
            }
            cache.clear();
            return model.forward(sequence.narrow(1, total - window, window), &cache);
        };
        const Tensor logits = run_model();
        const Tensor next = sample_next(logits.select(1, logits.size(1) - 1), options);
        sequence = cat({sequence, next}, 1);
        if (options.on_token && !options.on_token(next)) {
            break;
        }
    }
    model.train(was_training);
    return sequence;
}

void load_gpt2_weights(GPT& model, const std::filesystem::path& path) {
    const io::TensorMap stored = io::load_safetensors(path, io::LoadMode::Mapped);
    const StateDict expected = model.state_dict();
    StateDict translated;
    for (const auto& [stored_name, tensor] : stored) {
        const std::string name = translated_gpt2_name(stored_name);
        if (expected.contains(name)) {
            translated.emplace(name, stored_transposed(name) ? tensor.T().contiguous() : tensor);
        }
    }
    model.load_state_dict(translated);
}

} // namespace kael::nn
