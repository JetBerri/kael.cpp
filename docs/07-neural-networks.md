# 07. Neural networks and training

This chapter builds everything between a differentiable tensor and a trained classifier: modules with named parameters, layers, weight initialization, loss functions, optimizers, learning rate schedules, gradient clipping, datasets, a batching data loader and an MNIST reader. It ends with `examples/mnist_mlp.cpp`, a two-layer perceptron that reaches about 97% test accuracy on MNIST in five epochs on a CPU. It maps to milestone v0.1 in the README: "train a model on MNIST on CPU, end to end".

| File | Role |
|------|------|
| `include/kael/nn/module.h`, `src/nn/module.cpp` | `Module`, `UnaryModule`, parameter registry, `state_dict` |
| `include/kael/nn/linear.h` | `Linear` |
| `include/kael/nn/activation.h` | `ReLU`, `GELU`, `Sigmoid`, `Tanh` |
| `include/kael/nn/dropout.h` | `Dropout` |
| `include/kael/nn/embedding.h` | `Embedding` |
| `include/kael/nn/layernorm.h` | `LayerNorm` |
| `include/kael/nn/sequential.h` | `Sequential`, `Flatten` |
| `src/nn/layers.cpp` | Constructors and `forward` of every layer above |
| `include/kael/nn/functional.h`, `src/nn/functional.cpp` | Stateless functions: `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `dropout` |
| `include/kael/nn/init.h`, `src/nn/init.cpp` | `fans_of`, uniform, normal, constant, Xavier and Kaiming initializers |
| `include/kael/nn/loss.h`, `src/nn/loss.cpp` | `Reduction`, `mse_loss`, `nll_loss`, `cross_entropy`, `binary_cross_entropy_with_logits` |
| `include/kael/optim/optimizer.h` | `Optimizer` base class |
| `include/kael/optim/sgd.h`, `include/kael/optim/adam.h` | `SGD`, `Adam`, `AdamW` |
| `include/kael/optim/scheduler.h` | `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, `WarmupCosineLR` |
| `include/kael/optim/clip.h` | `clip_grad_norm_` |
| `src/optim/optimizer.cpp` | All optimizer, scheduler and clipping code |
| `include/kael/data/dataset.h`, `include/kael/data/dataloader.h`, `src/data/dataset.cpp` | `Example`, `Dataset`, `TensorDataset`, `DataLoader` |
| `include/kael/data/mnist.h`, `src/data/mnist.cpp` | IDX parser and `load_mnist` |
| `examples/mnist_mlp.cpp` | End-to-end MNIST training |
| `tests/test_training.cpp` | Small end-to-end training runs that must converge |

## Prerequisites

- [05. Operations](05-operations.md): `matmul`, `softmax`, `log_softmax`, `gather`, `index_select`, the fused `cross_entropy_losses` and `standardize`.
- [06. Autograd](06-autograd.md): `requires_grad_`, `backward`, `zero_grad`, `NoGradGuard`, the in-place grad rules and the version check.

## From autograd to training

A training step has four parts.

```mermaid
flowchart LR
    B[batch from DataLoader] --> F[forward: model]
    F --> L[loss]
    L --> Z[zero_grad]
    Z --> BW[backward]
    BW --> S[optimizer.step under NoGradGuard]
    S --> B
```

Autograd already provides the gradients. The rest of the chapter adds the bookkeeping around them: a way to collect every trainable tensor of a model (`Module`), a set of layers with sensible defaults (`Linear`, `LayerNorm`, ...), scalar objectives (losses), update rules that read `grad()` and write parameters in place (optimizers), and a way to feed batches (`DataLoader`).

## Modules

### Parameters and children

**Parameter** is the name for a leaf tensor that requires grad and that a module registers by name. A module owns a list of named parameters and a list of named child modules.

From `include/kael/nn/module.h`:

```cpp
class Module {
public:
    Tensor register_parameter(std::string name, Tensor parameter);

    template <typename M>
    std::shared_ptr<M> register_module(std::string name, std::shared_ptr<M> m) {
        add_child(std::move(name), m);
        return m;
    }

    [[nodiscard]] std::vector<Tensor> parameters() const;
    [[nodiscard]] std::vector<NamedTensor> named_parameters() const;
    ...
private:
    std::vector<NamedTensor> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};
```

`register_parameter` calls `requires_grad_()` on the tensor and stores the handle. Because `Tensor` is a handle (see [ADR 0002](adr/0002-tensor-as-handle.md)), the copy in `parameters_` and the copy in the layer's own member (`weight_`) share one `TensorImpl`. An optimizer that updates the copy it received from `parameters()` updates the layer.

`collect_parameters` walks the tree depth first and builds dotted names: a `Linear` registered as `"0"` inside a `Sequential` contributes `0.weight` and `0.bias`. `named_parameters()` then removes duplicates by `TensorImpl` address, so a parameter shared between two places (tied weights) appears once and the optimizer updates it once (`SharedParameterIsListedOnce`).

Modules are not copyable. Copying one would copy handles, and two "independent" models would share every weight.

`UnaryModule` adds the single-input interface used by every layer in this chapter:

```cpp
class UnaryModule : public Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input) { return forward(input); }
};
```

### Training mode

`train(bool)` sets `training_` and recurses into children; `eval()` is `train(false)`. Only `Dropout` reads the flag in this chapter.

### Other tree operations

| Member | Behavior |
|--------|----------|
| `zero_grad()` | `zero_grad()` on every unique parameter |
| `num_parameters()` | Sum of `numel()` over unique parameters |
| `to(Device)` | Under `NoGradGuard`, moves each parameter's storage with `move_to_` and moves its gradient too. The `TensorImpl` stays the same, so optimizers created earlier still see the parameters |
| `state_dict()` | `std::map<std::string, Tensor>` of `parameter.detach()` for every registered name, duplicates included |
| `load_state_dict(state)` | Validates every parameter name and shape first, then copies under `NoGradGuard` |

`state_dict` entries are detached views: they share storage with the parameters (`StateDictEntriesAreDetached`) but carry no autograd metadata, so saving them does not keep a graph alive. `load_state_dict` checks all names and shapes before it writes anything, so a mismatch leaves the model untouched. Keys in the state dict that match no parameter are ignored.

kael has no buffer registry. Modules hold only parameters, and non-trainable configuration (the `eps` of `LayerNorm`, the `p` of `Dropout`) lives in ordinary members. No layer in this chapter has running statistics, so nothing else needs to be saved.

## Layers

### Linear

`Linear(in_features, out_features, bias = true, dtype = DType::F32)` stores `weight` with shape `[out, in]` and an optional `bias` with shape `[out]`. The layout matches PyTorch, which makes loading PyTorch weights a direct copy.

From `src/nn/layers.cpp`:

```cpp
Tensor Linear::forward(const Tensor& input) {
    const Tensor product = matmul(input, weight_.T());
    return bias_ ? product + *bias_ : product;
}
```

For an input `[N, in]`, `matmul` with `weight.T()` (`[in, out]`) gives `[N, out]` and the bias broadcasts over rows. `matmul` also accepts a 1-D input `[in]` and returns `[out]` (`NnLinear.OutputShapes`). The backward pass needs no layer-specific code: `MatmulBackward`, `TransposeBackward` and `AddBackward` produce the gradients, and `sum_to_shape` sums the bias gradient over the batch.

The default initialization reproduces PyTorch's.

From `src/nn/layers.cpp`:

```cpp
Tensor default_linear_weight(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor weight = empty({out_features, in_features}, dtype);
    init::kaiming_uniform_(weight, std::sqrt(5.0));
    return weight;
}

Tensor default_linear_bias(std::int64_t in_features, std::int64_t out_features, DType dtype) {
    Tensor bias = empty({out_features}, dtype);
    const double limit = in_features > 0 ? 1.0 / std::sqrt(static_cast<double>(in_features)) : 0.0;
    return init::uniform_(bias, -limit, limit);
}
```

`kaiming_uniform_` with `a = sqrt(5)` has gain `sqrt(2 / (1 + 5)) = sqrt(1/3)` and bound `gain * sqrt(3 / fan_in) = 1 / sqrt(fan_in)`. For the MNIST layer `Linear(784, 128)` both weight and bias are uniform in `[-1/28, 1/28]`, about `[-0.0357, 0.0357]` (`NnLinear.DefaultInitBoundsMatchPyTorch`).

### Activations

`ReLU`, `GELU(approximate = false)`, `Sigmoid` and `Tanh` are stateless `UnaryModule` wrappers around `kael::nn::functional`, which forwards to the ops of [05. Operations](05-operations.md). They exist as modules so they can sit inside a `Sequential`. `GELU(true)` selects the tanh approximation used by GPT-2.

### Dropout

Dropout zeroes each element with probability `p` during training. kael implements inverted dropout: the kept elements are divided by `1 - p` at training time, so the expected value of each output equals its input and evaluation needs no rescaling.

From `src/nn/functional.cpp`:

```cpp
Tensor dropout(const Tensor& input, double p, bool training) {
    KAEL_CHECK_VALUE(p >= 0.0 && p < 1.0, "dropout probability must be in [0, 1), got " << p);
    if (!training || p == 0.0) {
        return input;
    }
    const double keep = 1.0 - p;
    return input * bernoulli(input.shape(), keep, input.dtype()).to(input.device()) / keep;
}
```

With `p = 0.2` and an input of ones, about 80% of the outputs are `1.25` and the rest are `0`; the mean stays `1.0` (`NnDropout.TrainKeepsProportionAndExpectedValue`). `p = 1` is rejected because `1 / keep` would divide by zero. The mask is drawn on the CPU from the global generator and moved to the input's device. The gradient flows through `MulBackward` and `DivBackward`, so dropped elements get a zero gradient without extra code.

### Embedding

`Embedding(count, dimension)` is a lookup table: `weight` has shape `[count, dimension]`, initialized from `N(0, 1)` like PyTorch.

From `src/nn/layers.cpp`:

```cpp
Tensor Embedding::forward(const Tensor& indices) {
    Shape output_shape = indices.shape();
    output_shape.push_back(weight_.size(1));
    return index_select(weight_, 0, indices.flatten()).reshape(output_shape);
}
```

Indices of shape `[2, 3]` produce `[2, 3, dimension]`. The backward of `index_select` is `index_add_` into a zero tensor, so a token that appears twice in a batch receives the sum of both gradients (`NnEmbedding.BackwardAddsRepeatedIndices`).

### LayerNorm

`LayerNorm(normalized_shape, eps = 1e-5)` normalizes the trailing dimensions to zero mean and unit variance, then applies a learned per-element scale and shift.

```text
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

`mean` and the biased `var` run over the last `normalized_shape.size()` dimensions. `weight` starts at ones and `bias` at zeros, so a fresh layer is a pure standardization.

From `src/nn/layers.cpp`:

```cpp
Tensor LayerNorm::forward(const Tensor& input) {
    const std::size_t count = normalized_shape_.size();
    KAEL_CHECK_VALUE(static_cast<std::size_t>(input.dim()) >= count &&
                         Shape(input.shape().end() - static_cast<std::ptrdiff_t>(count),
                               input.shape().end()) == normalized_shape_,
                     "LayerNorm over " << to_string(normalized_shape_) << " got input shape "
                                       << to_string(input.shape()));
    return kael::standardize(input, static_cast<std::int64_t>(count), eps_) * weight_ + bias_;
}
```

The normalization itself is the fused `standardize` op with its own `StandardizeBackward` formula, derived in [06. Autograd](06-autograd.md). The scale and shift use ordinary broadcasting ops. Worked example over one row `x = [1, 2, 3, 4]`: mean `2.5`, biased variance `1.25`, `sigma = sqrt(1.25 + 1e-5) = 1.11804`, output `[-1.3416, -0.4472, 0.4472, 1.3416]`.

### Sequential and Flatten

`Sequential` registers each layer under its position (`"0"`, `"1"`, ...) and applies them in order.

From `src/nn/layers.cpp`:

```cpp
void Sequential::add(std::shared_ptr<UnaryModule> layer) {
    layers_.push_back(register_module(std::to_string(layers_.size()), std::move(layer)));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}
```

`Flatten(start_dim = 1, end_dim = -1)` calls `input.flatten(start_dim, end_dim)`, which turns MNIST images `[N, 1, 28, 28]` into `[N, 784]` and keeps the batch dimension (`NnFlatten.KeepsBatchDimension`).

## Weight initialization

The scale of the initial weights decides whether activations and gradients keep a stable magnitude through depth. If each layer multiplies the variance of its input by a factor `c`, a network of depth `L` scales it by `c^L`. The standard schemes choose the weight variance so that `c` is close to 1.

`fans_of(weight)` computes the two quantities the schemes need. For a weight of shape `[out, in, k1, k2, ...]` the receptive field is `k1 * k2 * ...`.

From `src/nn/init.cpp`:

```cpp
Fans fans_of(const Tensor& weight) {
    KAEL_CHECK_VALUE(weight.dim() >= 2, ...);
    const std::int64_t receptive_field = weight.numel() / (weight.size(0) * weight.size(1));
    return {weight.size(1) * receptive_field, weight.size(0) * receptive_field};
}
```

| Function | Distribution | Parameter |
|----------|--------------|-----------|
| `uniform_(t, low, high)` | `U(low, high)` | |
| `normal_(t, mean, std)` | `N(mean, std^2)` | |
| `zeros_`, `ones_`, `constant_(t, value)` | constant | |
| `xavier_uniform_(t, gain = 1)` | `U(-b, b)` | `b = gain * sqrt(6 / (fan_in + fan_out))` |
| `kaiming_uniform_(t, a = 0)` | `U(-b, b)` | `b = gain(a) * sqrt(3 / fan_in)` |
| `kaiming_normal_(t, a = 0)` | `N(0, s^2)` | `s = gain(a) / sqrt(fan_in)` |

`gain(a) = sqrt(2 / (1 + a^2))` is the leaky ReLU gain; `a = 0` gives `sqrt(2)`, the ReLU gain. A uniform distribution on `[-b, b]` has variance `b^2 / 3`, so the uniform and normal Kaiming variants have the same variance `gain^2 / fan_in`. Xavier balances forward and backward variance with `2 / (fan_in + fan_out)`; Kaiming keeps the forward variance for ReLU networks, where half the units are zero.

Worked numbers from `NnInit.XavierAndKaimingLimits` for a weight `[100, 200]` (`fan_in = 200`, `fan_out = 100`):

```text
xavier bound  = sqrt(6 / 300)               = 0.1414
kaiming bound = sqrt(2) * sqrt(3 / 200)     = 0.1732
```

Every initializer runs under `NoGradGuard` and writes in place with `copy_`, `zero_` or `fill_`. Without the guard, the in-place rules of [06. Autograd](06-autograd.md) would reject writes to a parameter that requires grad (`NnInit.InitWorksOnParameterRequiringGrad`). The random values come from `rand` and `randn`, so `manual_seed` makes initialization reproducible.

## Loss functions

Every loss takes a `Reduction`: `Mean` (default), `Sum` or `None` (per-element or per-sample losses).

### Mean squared error

`mse_loss(prediction, target)` is `reduce((prediction - target)^2)`. Its gradient with mean reduction is `2 (prediction - target) / n`; autograd derives it from `SubBackward`, `MulBackward` and `MeanBackward`.

### Negative log-likelihood and cross entropy

`nll_loss(log_probabilities, targets)` expects `[N, C]` log-probabilities and `[N]` class indices of dtype `I64`, picks `log_probabilities[i, targets[i]]` with `gather` and negates it.

`cross_entropy(logits, targets)` is `nll_loss(log_softmax(logits, 1), targets)` mathematically, but it calls the fused `cross_entropy_losses` op. For one row with logits `x` and target `t`:

```text
loss = -log(softmax(x)[t]) = logsumexp(x) - x[t]
```

The fused op computes this per row after subtracting the row maximum, so large logits do not overflow `exp`. It records a single `CrossEntropyBackward` node whose gradient is `softmax(x) - onehot(t)` per row. On devices other than the CPU it falls back to `-gather(log_softmax(logits, 1), ...)`. Worked example (`NnLoss.CrossEntropyWorkedExample`): logits `[2, 1, 0.1]`, target 0, loss `0.4170`. With uniform logits over `C` classes the loss is `log(C)`, which is why a fresh classifier on 10 classes starts near `2.30` (`NnLoss.UniformPredictionGivesLogOfClasses`).

Both losses validate targets: shape `[N]`, dtype `I64`, every value in `[0, C)`. The error names the offending value ("target 10 is out of bounds for 10 classes"). Without the check an out-of-range target would read outside the row.

### Binary cross entropy with logits

For a logit `x` and a target `t` in `[0, 1]`, binary cross entropy on `p = sigmoid(x)` is `-(t log p + (1 - t) log(1 - p))`. Computing `sigmoid` first and then `log` overflows or returns `log(0)` for large `|x|`. Substituting `log sigmoid(x) = -log(1 + exp(-x))` and rearranging gives a form that never exponentiates a positive number:

```text
loss = max(x, 0) - x * t + log(1 + exp(-|x|))
```

From `src/nn/loss.cpp`:

```cpp
Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& targets,
                                        Reduction reduction) {
    const Tensor losses =
        maximum(logits, 0) - logits * targets + log1p(kael::exp(-kael::abs(logits)));
    return reduce(losses, reduction);
}
```

Worked values with `Reduction::None`:

| `x` | `t` | loss |
|-----|-----|------|
| 0 | 1 | `log 2 = 0.6931` |
| 2 | 1 | `2 - 2 + log1p(e^-2) = 0.1269` |
| -2 | 0 | `0 - 0 + log1p(e^-2) = 0.1269` |
| 1000 | 1 | `0` |

The last row is `NnLoss.BinaryCrossEntropyIsStableForLargeLogits`. kael has no `binary_cross_entropy` on probabilities; pass logits.

## Optimizers

### The base class

From `include/kael/optim/optimizer.h`:

```cpp
class Optimizer {
public:
    Optimizer(std::vector<Tensor> parameters, double learning_rate);
    void step();
    void zero_grad();
    [[nodiscard]] double learning_rate() const noexcept { return learning_rate_; }
    void set_learning_rate(double learning_rate);
protected:
    [[nodiscard]] std::vector<Tensor>& parameters() noexcept { return parameters_; }
private:
    virtual void update() = 0;
    ...
};
```

The constructor rejects a negative learning rate and any parameter that is not a leaf requiring grad. `step()` is a template method: it installs a `NoGradGuard` and calls the subclass's `update()`. The guard is what allows `parameter.sub_(...)` on a leaf that requires grad, and it keeps the update from recording a graph. The update is in place, so the parameter keeps its `TensorImpl` and storage, stays a leaf and stays registered in its module (`Optimizer.StepCreatesNoGraphAndKeepsTheSameTensor`). Parameters with no gradient are skipped (`Optimizer.ParametersWithoutGradAreSkipped`), which matters for layers a forward pass did not use.

### SGD with momentum

`SgdOptions` has `learning_rate` (default `1e-3`), `momentum`, `weight_decay` and `nesterov`. The update for gradient `g`, parameter `w`, momentum `mu`, weight decay `lambda` and learning rate `lr`:

```text
g = g + lambda * w                    (if weight_decay > 0)
v = g                                 (first step)
v = mu * v + g                        (later steps)
d = g + mu * v  if nesterov, else v
w = w - lr * d
```

From `src/optim/optimizer.cpp`:

```cpp
if (options_.momentum > 0.0) {
    std::optional<Tensor>& velocity = velocities_[i];
    if (velocity) {
        velocity->mul_(options_.momentum).add_(gradient);
    } else {
        velocity = gradient.clone();
    }
    gradient = options_.nesterov ? gradient + *velocity * options_.momentum : *velocity;
}
parameter.sub_(gradient * learning_rate());
```

This is PyTorch's formulation: the velocity accumulates raw gradients and the learning rate multiplies at the end, so changing the learning rate mid-training does not rescale the stored velocity. Worked example: `w = 1`, loss `2w` (so `g = 2` every step), `lr = 0.1`, `mu = 0.9`.

```text
step 1: v = 2                 w = 1    - 0.1*2    = 0.8
step 2: v = 0.9*2    + 2 = 3.8   w = 0.8  - 0.1*3.8  = 0.42
step 3: v = 0.9*3.8  + 2 = 5.42  w = 0.42 - 0.1*5.42 = -0.122
```

The velocity grows towards `g / (1 - mu) = 20`, so momentum `0.9` amplifies a constant gradient up to ten times. Nesterov without positive momentum is rejected.

### Adam and AdamW

Adam keeps two moving averages per parameter: the first moment `m` (mean of gradients) and the second moment `v` (mean of squared gradients). Both start at zero, so early averages are biased towards zero; dividing by `1 - beta^t` corrects that.

```text
t = t + 1
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

From `src/optim/optimizer.cpp`:

```cpp
first.mul_(options_.beta1).add_(gradient * (1.0 - options_.beta1));
second.mul_(options_.beta2).add_(gradient * gradient * (1.0 - options_.beta2));
const Tensor corrected_first = first / first_correction;
const Tensor corrected_second = second / second_correction;
parameter.sub_(corrected_first / (kael::sqrt(corrected_second) + options_.epsilon) *
               learning_rate());
```

On the first step `m_hat = g` and `v_hat = g^2`, so each element moves by `lr * g / (|g| + eps)`, which is `lr` in the direction of `-sign(g)` regardless of the gradient's scale (`Optimizer.AdamFirstStepMovesByLearningRate`). For `w = 1` and loss `w^2` with `lr = 0.1` the first three iterates are `0.9`, `0.800412`, `0.701586`.

`AdamOptions` defaults: `learning_rate = 1e-3`, `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`, `weight_decay = 0`. `Adam` and `AdamW` share `AdamBase` and differ only in `WeightDecayMode`.

From `src/optim/optimizer.cpp`:

```cpp
if (options_.weight_decay > 0.0) {
    if (mode_ == WeightDecayMode::Coupled) {
        gradient = gradient + parameter * options_.weight_decay;
    } else {
        parameter.mul_(1.0 - learning_rate() * options_.weight_decay);
    }
}
```

- **Coupled** decay (`Adam`) adds `lambda * w` to the gradient. The decay term then passes through the adaptive scaling, so weights with large gradient history decay less.
- **Decoupled** decay (`AdamW`) shrinks the weight directly by `lr * lambda` and leaves the moments untouched. `AdamWOptions` sets `weight_decay = 0.01` by default.

With a zero gradient, one `AdamW` step at the defaults multiplies the weight by `1 - 1e-3 * 0.01`: `1.0` becomes `0.99999`.

`step_count_` is shared by all parameters of one optimizer and increments once per `step()`, as in PyTorch with a single parameter group.

### Learning rate schedulers

A scheduler owns a reference to an optimizer and rewrites its learning rate. The base class records the optimizer's learning rate at construction as `base_lr` and keeps a step counter.

From `src/optim/optimizer.cpp`:

```cpp
void LRScheduler::step() {
    ++step_count_;
    apply_schedule();
}

void LRScheduler::apply_schedule() {
    optimizer_.set_learning_rate(lr_at(step_count_));
}
```

Each subclass constructor calls `apply_schedule()`, so the learning rate for step 0 is in place before the first `optimizer.step()`. Call `scheduler.step()` after each `optimizer.step()`.

| Scheduler | `lr_at(step)` |
|-----------|---------------|
| `StepLR(step_size, gamma)` | `base * gamma^(floor(step / step_size))` |
| `CosineAnnealingLR(total, min_lr = 0)` | `min + (base - min) * (1 + cos(pi * min(step, total) / total)) / 2` |
| `LinearWarmup(warmup)` | `base * min(1, step / warmup)` |
| `WarmupCosineLR(warmup, total, min_lr = 0)` | linear warmup until `warmup`, then cosine from `base` to `min_lr` over the remaining `total - warmup` steps |

Worked values:

```text
CosineAnnealingLR, base 0.001, total 10, after 5 steps:
  0.5 * 0.001 * (1 + cos(pi/2)) = 0.0005

LinearWarmup, base 0.1, warmup 4, steps 0..5:
  0, 0.025, 0.05, 0.075, 0.1, 0.1
```

`LinearWarmup` and `WarmupCosineLR` set the learning rate to `0` at construction, so the first optimizer step after construction is a no-op update. The cosine schedules clamp the step at `total`, so the rate stays at `min_lr` afterwards. Invalid lengths (a non-positive step size or total, a warmup that does not fit in the total) throw (`Schedulers.InvalidLengthsThrow`).

### Gradient clipping

`clip_grad_norm_(parameters, max_norm)` treats all gradients as one vector, computes its L2 norm, and scales every gradient in place when the norm exceeds `max_norm`.

From `src/optim/optimizer.cpp`:

```cpp
const double total = std::sqrt(squared_total);
const double coefficient = max_norm / (total + 1e-6);
if (coefficient < 1.0) {
    for (const Tensor& parameter : parameters) {
        if (auto gradient = parameter.grad()) {
            gradient->mul_(coefficient);
        }
    }
}
return total;
```

`grad()` returns a handle to the stored gradient, so `mul_` on it updates the parameter's gradient. The function returns the norm before clipping, which is the value worth logging. Two gradients `[3]` and `[4]` have total norm `5`; with `max_norm = 1` each is multiplied by about `0.2`. Clipping preserves the direction of the full gradient, unlike per-element clamping. Call it between `backward()` and `optimizer.step()`.

## Data

### Dataset and TensorDataset

From `include/kael/data/dataset.h`:

```cpp
struct Example {
    Tensor input;
    Tensor target;
};

class Dataset {
public:
    [[nodiscard]] virtual std::int64_t size() const = 0;
    [[nodiscard]] virtual Example get(std::int64_t index) const = 0;
    [[nodiscard]] virtual Example get_batch(const Tensor& indices) const;
};
```

The default `get_batch` calls `get` for each index and `stack`s the results, which works for any dataset. `TensorDataset(inputs, targets)` overrides it with two `index_select` calls along dimension 0, one copy per batch instead of one per example. Its constructor requires equal first dimensions.

### DataLoader

`DataLoader(dataset, {.batch_size, .shuffle, .drop_last})` is a range of `Example` batches. It holds a reference to the dataset; the dataset must outlive it.

From `src/data/dataset.cpp`:

```cpp
DataLoader::Iterator DataLoader::begin() const {
    const std::int64_t size = dataset_.size();
    Tensor order =
        options_.shuffle ? randperm(size) : arange(0, static_cast<double>(size), 1.0, DType::I64);
    return Iterator(this, std::move(order), 0);
}
```

Each call to `begin()` draws a fresh permutation from the global generator, so every epoch of a range-`for` sees a new order, and `manual_seed` makes the sequence of orders reproducible (`DataLoader.ShuffleIsReproducibleWithSeedAndChangesBetweenEpochs`). The iterator compares by batch index only; `end()` is the iterator at `num_batches()`. Dereferencing builds the batch on first access and caches it until `++`.

```text
num_batches = drop_last ? size / batch_size : ceil(size / batch_size)

60000 examples, batch 64:  938 batches, the last one has 60000 - 937*64 = 32 examples
                           937 batches with drop_last
```

### MNIST and the IDX format

MNIST ships as four IDX files. `scripts/download_mnist.sh` fetches them into `data/mnist` (or the directory given as its argument), checks their MD5 sums and decompresses them. An IDX file is a big-endian header followed by raw bytes.

```text
offset  bytes  content
0       4      magic: 0x00000803 (images, 3 dims) or 0x00000801 (labels, 1 dim)
4       4      size of dim 0 (count)
8       4      size of dim 1 (rows)      images only
12      4      size of dim 2 (columns)   images only
16      ...    one unsigned byte per pixel or label, row-major
```

The magic encodes the element type in its third byte (`0x08`, unsigned byte) and the number of dimensions in its fourth.

From `src/data/mnist.cpp`:

```cpp
std::uint32_t read_big_endian(std::ifstream& file, const std::filesystem::path& path) {
    std::array<unsigned char, 4> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    KAEL_CHECK(file.good(), "IDX file " << path << " is truncated in its header");
    return (std::uint32_t{bytes[0]} << 24U) | (std::uint32_t{bytes[1]} << 16U) |
           (std::uint32_t{bytes[2]} << 8U) | std::uint32_t{bytes[3]};
}
```

Assembling the integer from bytes works on any host byte order, unlike reading a `uint32_t` and swapping. `read_payload` compares the number of bytes left in the file with `numel(shape)` before allocating. A truncated file, trailing garbage or a header that declares `0xFFFFFFFF` in every dimension all throw without allocating gigabytes (`Idx.TruncatedDataThrows`, `Idx.TrailingDataThrows`, `Idx.HugeDeclaredSizeThrowsBeforeAllocating`).

`read_idx_images` returns `F32` of shape `[N, 1, 28, 28]` with pixels scaled to `[0, 1]`. `read_idx_labels` returns `I64` of shape `[N]`. `load_mnist(directory, MnistPart::Train)` reads `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` (`t10k-*` for `MnistPart::Test`), checks that the counts match, and standardizes the images with the conventional MNIST statistics `mean = 0.1307`, `std = 0.3081`. A black pixel becomes `-0.4242` and a white pixel `2.8215`.

## The MNIST training loop

`examples/mnist_mlp.cpp` puts the chapter together. The model is `784 -> hidden -> 10` with a ReLU, trained with `AdamW` at its defaults (`lr = 1e-3`, `weight_decay = 0.01`) on batches of 64 for 5 epochs.

From `examples/mnist_mlp.cpp`:

```cpp
kael::nn::Sequential model{
    std::make_shared<kael::nn::Flatten>(), std::make_shared<Linear>(784, options.hidden),
    std::make_shared<kael::nn::ReLU>(), std::make_shared<Linear>(options.hidden, 10)};
model.to(options.device);
kael::optim::AdamW optimizer(model.parameters());
const kael::data::DataLoader loader(train_set, {.batch_size = kBatchSize, .shuffle = true});
```

`model.to` runs before the optimizer is built. `to` moves storage inside the same `TensorImpl`, so the handles the optimizer holds stay valid either way, but `AdamBase` allocates its moment buffers with `zeros_like(parameter)` in its constructor. Moving the model afterwards would leave those buffers on the old device.

One training epoch:

```cpp
for (const auto& [images, labels] : loader) {
    const Tensor loss = kael::nn::cross_entropy(model(images.to(device)), labels.to(device));
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    ...
}
```

`zero_grad` sits between the forward pass and `backward`, which is the latest point it can run: gradients accumulate, so the previous step's gradients must be cleared before this step's arrive.

Evaluation switches to eval mode, disables recording, and counts correct predictions in batches of 1000:

```cpp
model.eval();
const kael::autograd::NoGradGuard guard;
std::int64_t correct = 0;
for (const auto& [images, labels] : kael::data::DataLoader(dataset, {.batch_size = 1000})) {
    const Tensor predictions = kael::argmax(model(images.to(device)), 1);
    correct += kael::sum(predictions == labels.to(device)).item<std::int64_t>();
}
```

`NoGradGuard` saves memory and time: no nodes and no saved activations. `eval()` does not change this model's output (it has no dropout) but is the correct habit.

Arguments: a positional data directory (default `data/mnist`), `--device cpu|cuda` and `--hidden N` (default 128). The seed is fixed at `42`.

```bash
scripts/download_mnist.sh
cmake -S . -B build -DKAEL_BUILD_EXAMPLES=ON
cmake --build build --target mnist_mlp
build/examples/mnist_mlp data/mnist
```

A run of the Debug build in `build/` on an Apple M5 Pro printed these test accuracies:

```text
epoch 1 test accuracy 96% in 7.30869 s
epoch 2 test accuracy 96.73% in 7.14235 s
epoch 3 test accuracy 97.35% in 7.33984 s
epoch 4 test accuracy 97.35% in 7.39241 s
epoch 5 test accuracy 97.42% in 8.22106 s
```

The first logged loss is `2.40`, close to `log(10) = 2.30` as expected from a fresh classifier. No unit test runs the example, because CI does not download MNIST. `scripts/test_cuda.sh` runs it on CPU and on CUDA (hidden 128 and 1024) and fails if the final accuracy is below `minimum_mnist_accuracy="97"`. Timings depend on the machine and build type.

`examples/mnist.py` is the same program through the Python bindings, described in [10. Python bindings](10-python-bindings.md).

## Edge cases and pitfalls

- **Forgetting `zero_grad`.** Gradients accumulate across `backward()` calls. Skipping `zero_grad` sums the gradients of every past batch.
- **Stepping between forward and backward.** `optimizer.step()` updates parameters in place and bumps their version. A graph built before the step that saved a parameter (the second `Linear` of a stack saves its weight for the input gradient) fails at `backward()` with the in-place version error.
- **Writing to parameters outside an optimizer.** In-place writes to a parameter throw while grad mode is on. Use `NoGradGuard`, as `init`, `load_state_dict` and `Module::to` do.
- **Dropout probability.** `p` must be in `[0, 1)`. `Dropout(1.0)` and `Dropout(-0.1)` throw.
- **Targets.** Class targets must be `I64`, 1-D, as long as the batch, and inside `[0, C)`. A float label tensor is a type error, not a silent cast.
- **Large logits.** Use `cross_entropy` and `binary_cross_entropy_with_logits` on raw scores. Both are written to avoid overflow; composing `softmax` then `log` is not.
- **Warmup schedulers start at zero.** The first update after constructing `LinearWarmup` or `WarmupCosineLR` uses learning rate `0`.
- **DataLoader lifetime.** `DataLoader` stores `const Dataset&`. Constructing it from a temporary dataset leaves a dangling reference.
- **Tied parameters in `state_dict`.** A tensor registered under two names appears once in `parameters()` but twice in `state_dict()`. `load_state_dict` copies both entries into the same storage, so the second wins.
- **Empty batches.** `cross_entropy` on zero rows skips the target range check and its gradient is an empty tensor (`CrossEntropyFusedGraph.GradientOfAnEmptyBatchIsEmpty`).
- **Malformed IDX files.** Wrong magic, short header, short or long payload and missing files all throw `kael::Error` with the path in the message.

## How it is tested

| Test file | Suites | What it proves |
|-----------|--------|----------------|
| `tests/test_nn_module.cpp` | `NnModule`, `NnSequential` | Recursive collection, dotted names in registration order, registered parameters require grad, training flag propagation, `zero_grad`, `state_dict` round trip between two models, detached entries, shape and name checks in `load_state_dict`, tied parameters listed once, null child rejected, `Sequential` naming and order |
| `tests/test_nn_layers.cpp` | `NnInit`, `NnLinear`, `NnActivation`, `NnDropout`, `NnLayerNorm`, `NnEmbedding`, `NnFlatten`, `NnLoss`, `CrossEntropyFusedGraph` | Fans, initializer ranges and moments, PyTorch default bounds for `Linear`, layer outputs against PyTorch golden data, gradcheck for `Linear`, `LayerNorm`, `Embedding` and every loss, dropout statistics, stable BCE, target validation, the fused cross entropy node on contiguous and non-contiguous logits |
| `tests/test_optim.cpp` | `OptimGolden`, `Optimizer`, `ClipGradNorm`, `Schedulers` | SGD (plain, momentum, Nesterov, weight decay) and Adam, Adam with decay, AdamW trajectories match PyTorch after steps 1 and 5; Adam's first step size; in-place update keeps the tensor; skipping parameters without grad; clipping above and below the limit; every scheduler against PyTorch or hand values |
| `tests/test_data.cpp` | `Idx`, `TensorDataset`, `Dataset`, `DataLoader` | IDX parsing of synthetic files written by the test (including big-endian sizes and every malformed case), dataset indexing, default `get_batch` stacking, batch counts, order, `drop_last`, shuffle coverage and reproducibility |
| `tests/test_training.cpp` | `Training` | A `20 -> 64 -> 4` MLP overfits 32 random examples (initial loss near `log 4`, final loss below `0.02` after 200 Adam steps); `Linear(1, 1)` with SGD recovers `y = 3x + 2` within `0.05`; a `2 -> 8 -> 2` tanh MLP solves XOR; the same seed gives bit-identical losses and a different seed does not |

Golden data comes from `scripts/gen_golden.py`. `write_nn_fixtures` (seed `20261`) saves inputs and outputs of `torch.nn.functional.linear`, `layer_norm` over `(3, 4)`, `embedding`, and the four losses under every reduction into `tests/golden/nn/`. `write_optim_fixtures` runs each optimizer for five steps on the loss `sum(w * c) + 0.5 * sum(w^2)` and saves the parameter after steps 1 and 5, and records the learning rate sequence of `StepLR(step_size=3, gamma=0.5)` and `CosineAnnealingLR(T_max=10, eta_min=0.001)`, into `tests/golden/optim/`.

## Build it yourself

1. Write `Module` with `register_parameter`, `register_module`, `named_parameters` with dotted names and deduplication, `parameters`, `train`/`eval`, `zero_grad`. Checkpoint: `NnModule.ParametersAreCollectedRecursively`, `NamedParametersUseDottedNamesInOrder`, `SharedParameterIsListedOnce`.
2. Add `state_dict` and `load_state_dict` with validation before writing. Checkpoint: `NnModule.StateDict*` and `LoadStateDict*`.
3. Write `init`: `fans_of`, `uniform_`, `normal_`, the constant fills, Xavier and Kaiming, all under `NoGradGuard`. Checkpoint: `NnInit.*`.
4. Write `Linear` with the PyTorch default initialization. Checkpoint: `NnLinear.*`, including `MatchesPyTorch` and `Gradcheck`.
5. Write the activation modules, `Dropout`, `Flatten` and `Sequential`. Checkpoint: `NnActivation.*`, `NnDropout.*`, `NnFlatten.*`, `NnSequential.*`.
6. Write `LayerNorm` on top of `standardize` and `Embedding` on top of `index_select`. Checkpoint: `NnLayerNorm.*`, `NnEmbedding.*`.
7. Write the losses with `Reduction`. Checkpoint: `NnLoss.*` and `CrossEntropyFusedGraph.*`.
8. Write `Optimizer` with `step()` under `NoGradGuard`, then `SGD`. Checkpoint: `Training.LinearRegressionConverges` and `OptimGolden.SgdVariantsMatchPyTorch`.
9. Write `AdamBase`, `Adam` and `AdamW`. Checkpoint: `OptimGolden.AdamVariantsMatchPyTorch`, `Optimizer.AdamFirstStepMovesByLearningRate`, then all of `Training.*`.
10. Write the schedulers and `clip_grad_norm_`. Checkpoint: `Schedulers.*`, `ClipGradNorm.*`.
11. Write `Dataset`, `TensorDataset` and `DataLoader`. Checkpoint: `TensorDataset.*`, `Dataset.*`, `DataLoader.*`.
12. Write the IDX reader and `load_mnist`. Checkpoint: `Idx.*`.
13. Run `scripts/download_mnist.sh`, configure with `-DKAEL_BUILD_EXAMPLES=ON`, build `mnist_mlp` and run it. Checkpoint: the last line reports at least 97% test accuracy.

## Further reading

- Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks" (Xavier initialization).
- He, Zhang, Ren and Sun, 2015, the paper that introduced Kaiming initialization and PReLU.
- Kingma and Ba, "Adam: A Method for Stochastic Optimization".
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (AdamW) and "SGDR: Stochastic Gradient Descent with Warm Restarts" (cosine schedules).
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting".
- Ba, Kiros and Hinton, "Layer Normalization".
- LeCun, Cortes and Burges, the MNIST database page, which documents the IDX format.

Next: [08. CPU performance](08-cpu-performance.md)
