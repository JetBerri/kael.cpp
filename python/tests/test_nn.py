import math

import numpy as np
import pytest

import kael
from kael import nn, optim
from kael.nn import functional as F

torch = pytest.importorskip("torch")


def k_tensor(array, dtype=kael.float64):
    return kael.tensor(array, dtype=dtype)


def t_tensor(array):
    return torch.tensor(array, dtype=torch.float64)


def assert_close(actual, expected, tolerance=1e-9):
    np.testing.assert_allclose(
        np.array(actual.tolist()), expected.detach().numpy(), rtol=tolerance, atol=tolerance
    )


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 3)
        self.scale = nn.Parameter(kael.ones(1))
        self.activation = nn.ReLU()
        self.inner = nn.Sequential(nn.Linear(3, 2), nn.Tanh())
        self.plain = 5

    def forward(self, x):
        return self.inner(self.activation(self.first(x))) * self.scale


def test_attributes_register_automatically():
    model = Small()
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "scale",
        "first.weight",
        "first.bias",
        "inner.0.weight",
        "inner.0.bias",
    ]
    assert model.num_parameters() == 1 + 12 + 3 + 6 + 2
    assert model.plain == 5


def test_reassigning_an_attribute_replaces_it():
    model = Small()
    model.scale = 3
    assert "scale" not in dict(model.named_parameters())


def test_parameters_are_leaves_requiring_grad():
    for parameter in Small().parameters():
        assert parameter.requires_grad and parameter.is_leaf
        assert isinstance(parameter, kael.Tensor)


def test_module_needs_super_init():
    class Broken(nn.Module):
        def __init__(self):
            self.x = 1

    with pytest.raises(AttributeError):
        Broken()


def test_shared_parameters_appear_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2, bias=False)
            self.b = nn.Linear(2, 2, bias=False)
            self.b.weight = self.a.weight

    assert len(Tied().parameters()) == 1


def test_train_and_eval_propagate():
    model = Small()
    model.eval()
    assert not model.inner.training and not model.first.training
    model.train()
    assert model.inner.training


def test_call_runs_forward_and_trains():
    model = Small()
    output = model(kael.ones(2, 4))
    assert output.shape == (2, 2)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.zero_grad()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_state_dict_round_trip():
    model = Small()
    other = Small()
    other.load_state_dict(model.state_dict())
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        assert kael.equal(a.detach(), b.detach()), name


def test_state_dict_validates():
    model = Small()
    state = model.state_dict()
    state["scale"] = kael.zeros(9)
    with pytest.raises(ValueError):
        model.load_state_dict(state)
    del state["scale"]
    with pytest.raises(KeyError):
        model.load_state_dict(state)


def test_buffers_are_kept_in_state_dict_only():
    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", kael.ones(2))

    model = WithBuffer()
    assert model.parameters() == []
    assert "mask" in model.state_dict()


def test_linear_matches_torch():
    x = np.random.randn(5, 3)
    weight = np.random.randn(2, 3)
    bias = np.random.randn(2)
    layer = nn.Linear(3, 2, dtype=kael.float64)
    layer.load_state_dict({"weight": k_tensor(weight), "bias": k_tensor(bias)})
    reference = torch.nn.Linear(3, 2, dtype=torch.float64)
    reference.load_state_dict({"weight": t_tensor(weight), "bias": t_tensor(bias)})
    assert_close(layer(k_tensor(x)), reference(t_tensor(x)))


def test_linear_without_bias():
    layer = nn.Linear(3, 2, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1
    assert layer(kael.ones(1, 3)).shape == (1, 2)


def test_linear_init_range():
    layer = nn.Linear(100, 100)
    limit = 1 / math.sqrt(100)
    assert layer.weight.detach().abs().max().item() <= limit + 1e-6


def test_layer_norm_matches_torch():
    x = np.random.randn(2, 3, 4)
    reference = torch.nn.LayerNorm(4, dtype=torch.float64)
    layer = nn.LayerNorm(4, dtype=kael.float64)
    assert_close(layer(k_tensor(x)), reference(t_tensor(x)))
    with pytest.raises(ValueError):
        layer(kael.zeros(3, dtype=kael.float64))


def test_embedding_shape_and_values():
    layer = nn.Embedding(5, 3)
    indices = kael.tensor([[0, 4], [1, 1]])
    assert layer(indices).shape == (2, 2, 3)
    assert kael.equal(layer(indices)[0, 1], layer.weight.detach()[4])


def test_dropout_modes():
    layer = nn.Dropout(0.5)
    x = kael.ones(1000)
    output = layer(x)
    assert 300 < int((output == 0).sum().item()) < 700
    assert output.max().item() == 2
    layer.eval()
    assert kael.equal(layer(x), x)
    with pytest.raises(ValueError):
        F.dropout(x, 1.5)


def test_flatten_sigmoid_softmax():
    assert nn.Flatten()(kael.zeros(2, 3, 4)).shape == (2, 12)
    assert nn.Sigmoid()(kael.zeros(1)).item() == 0.5
    assert math.isclose(nn.Softmax()(kael.zeros(4))[0].item(), 0.25)


def test_module_list():
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    assert len(layers) == 2
    assert len(layers.parameters()) == 4
    assert layers[1] is list(layers)[1]


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_losses_match_torch(reduction):
    logits = np.random.randn(6, 4)
    targets = np.random.randint(0, 4, 6)
    assert_close(
        F.cross_entropy(k_tensor(logits), kael.tensor(targets), reduction),
        torch.nn.functional.cross_entropy(t_tensor(logits), torch.tensor(targets), reduction=reduction),
    )
    other = np.random.randn(6, 4)
    assert_close(
        F.mse_loss(k_tensor(logits), k_tensor(other), reduction),
        torch.nn.functional.mse_loss(t_tensor(logits), t_tensor(other), reduction=reduction),
    )
    probabilities = np.random.rand(6, 4)
    assert_close(
        F.binary_cross_entropy_with_logits(k_tensor(logits), k_tensor(probabilities), reduction),
        torch.nn.functional.binary_cross_entropy_with_logits(
            t_tensor(logits), t_tensor(probabilities), reduction=reduction
        ),
    )


def test_loss_validation():
    with pytest.raises(ValueError):
        F.nll_loss(kael.zeros(2), kael.tensor([0, 1]))
    with pytest.raises(ValueError):
        F.cross_entropy(kael.zeros(2, 2), kael.tensor([0]))
    with pytest.raises(TypeError):
        F.cross_entropy(kael.zeros(2, 2), kael.zeros(2))
    with pytest.raises(IndexError):
        F.cross_entropy(kael.zeros(2, 2), kael.tensor([0, 5]))
    with pytest.raises(ValueError):
        F.mse_loss(kael.zeros(1), kael.zeros(1), "bad")


OPTIMIZERS = [
    (lambda p: optim.SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1)),
    (
        lambda p: optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.01),
        lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.01),
    ),
    (lambda p: optim.Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
    (
        lambda p: optim.Adam(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1),
    ),
    (lambda p: optim.AdamW(p, lr=0.01), lambda p: torch.optim.AdamW(p, lr=0.01)),
]


@pytest.mark.parametrize("build_kael, build_torch", OPTIMIZERS)
def test_optimizers_match_torch(build_kael, build_torch):
    start = np.random.randn(3, 2)
    k = nn.Parameter(k_tensor(start))
    t = torch.nn.Parameter(t_tensor(start))
    k_optimizer = build_kael([k])
    t_optimizer = build_torch([t])
    for _ in range(25):
        k_optimizer.zero_grad()
        (k * k + kael.sin(k)).sum().backward()
        k_optimizer.step()
        t_optimizer.zero_grad()
        (t * t + torch.sin(t)).sum().backward()
        t_optimizer.step()
    assert_close(k.detach(), t.detach(), 1e-7)


def test_optimizer_skips_parameters_without_grad():
    parameter = nn.Parameter(kael.ones(2))
    optimizer = optim.SGD([parameter], lr=1.0)
    optimizer.step()
    assert parameter.detach().tolist() == [1.0, 1.0]


def test_optimizer_validates_learning_rate():
    with pytest.raises(ValueError):
        optim.SGD([], lr=-1)


def test_clip_grad_norm_matches_torch():
    gradients = np.random.randn(4, 3)
    k = nn.Parameter(k_tensor(gradients))
    t = torch.nn.Parameter(t_tensor(gradients))
    (k * k_tensor(gradients)).sum().backward()
    (t * t_tensor(gradients)).sum().backward()
    k_norm = optim.clip_grad_norm_([k], 0.5)
    t_norm = torch.nn.utils.clip_grad_norm_([t], 0.5)
    assert math.isclose(k_norm, t_norm.item(), rel_tol=1e-9)
    np.testing.assert_allclose(np.array(k.grad.tolist()), t.grad.numpy(), rtol=1e-5)
    assert optim.clip_grad_norm_([], 1.0) == 0.0


def run_schedule(scheduler, optimizer, steps):
    rates = []
    for _ in range(steps):
        rates.append(optimizer.lr)
        scheduler.step()
    return rates


def test_schedulers_match_torch():
    k_opt = optim.SGD([], lr=0.5)
    parameter = torch.nn.Parameter(torch.zeros(1))
    t_opt = torch.optim.SGD([parameter], lr=0.5)
    k_sched = optim.StepLR(k_opt, 3, 0.1)
    t_sched = torch.optim.lr_scheduler.StepLR(t_opt, 3, 0.1)
    for _ in range(10):
        assert math.isclose(k_opt.lr, t_opt.param_groups[0]["lr"], rel_tol=1e-9)
        k_sched.step()
        t_opt.step()
        t_sched.step()

    k_opt = optim.SGD([], lr=0.5)
    t_opt = torch.optim.SGD([parameter], lr=0.5)
    k_sched = optim.CosineAnnealingLR(k_opt, 8)
    t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, 8)
    for _ in range(9):
        assert math.isclose(k_opt.lr, t_opt.param_groups[0]["lr"], rel_tol=1e-9, abs_tol=1e-12)
        k_sched.step()
        t_opt.step()
        t_sched.step()


def test_warmup_cosine_shape():
    optimizer = optim.SGD([], lr=1.0)
    rates = run_schedule(optim.WarmupCosineLR(optimizer, 2, 6), optimizer, 7)
    assert rates[0] == 0.5
    assert rates[1] == 1.0
    assert rates[2] == 1.0
    assert rates[-1] < 1e-9
    assert rates[3] > rates[4] > rates[5]


def test_model_learns_xor():
    kael.manual_seed(3)
    x = kael.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = kael.tensor([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    optimizer = optim.AdamW(model.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
    assert kael.argmax(model(x), 1).tolist() == [0, 1, 1, 0]


def test_module_to_keeps_parameters_on_cpu():
    model = Small()
    parameter = model.first.weight
    assert model.to("cpu") is model
    assert model.first.weight is parameter
