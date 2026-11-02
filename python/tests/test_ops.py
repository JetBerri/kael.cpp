import numpy as np
import pytest

import kael

torch = pytest.importorskip("torch")

KAEL_TO_TORCH = {kael.float32: torch.float32, kael.float64: torch.float64}


def both(array, dtype=kael.float64):
    torch_dtype = KAEL_TO_TORCH[dtype]
    return kael.tensor(array, dtype=dtype), torch.tensor(array, dtype=torch_dtype)


def assert_same(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(
        np.array(actual.tolist(), dtype=float),
        expected.detach().numpy().astype(float),
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


UNARY = ["abs", "exp", "sin", "cos", "tanh", "sigmoid", "relu", "neg"]


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("dtype", [kael.float32, kael.float64])
def test_unary_ops_match_torch(name, dtype):
    k, t = both(np.random.randn(5, 4), dtype)
    torch_fn = getattr(torch, name)
    tolerance = 1e-5 if dtype == kael.float32 else 1e-10
    assert_same(getattr(kael, name)(k), torch_fn(t), tolerance)
    assert_same(getattr(k, name)(), torch_fn(t), tolerance)


@pytest.mark.parametrize("name", ["log", "sqrt", "log1p"])
def test_positive_domain_ops_match_torch(name):
    k, t = both(np.random.rand(5, 4) + 0.1)
    assert_same(getattr(kael, name)(k), getattr(torch, name)(t))


@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_matches_torch(approximate):
    k, t = both(np.random.randn(5, 4))
    expected = torch.nn.functional.gelu(t, approximate="tanh" if approximate else "none")
    assert_same(kael.gelu(k, approximate), expected)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 1)), ((3,), ())],
)
def test_binary_operators_match_torch_with_broadcast(shape_a, shape_b):
    ka, ta = both(np.random.randn(*shape_a) + 3)
    kb, tb = both(np.random.randn(*shape_b) + 3)
    assert_same(ka + kb, ta + tb)
    assert_same(ka - kb, ta - tb)
    assert_same(ka * kb, ta * tb)
    assert_same(ka / kb, ta / tb)
    assert_same(ka // kb, ta // tb)
    assert_same(ka**2, ta**2)


@pytest.mark.parametrize("op", ["__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"])
def test_comparisons_match_torch(op):
    ka, ta = both(np.random.randint(0, 3, (4, 4)).astype(float))
    kb, tb = both(np.random.randint(0, 3, (4, 4)).astype(float))
    result = getattr(ka, op)(kb)
    assert result.dtype == kael.bool
    assert result.tolist() == getattr(ta, op)(tb).tolist()


@pytest.mark.parametrize("name", ["sum", "mean", "prod"])
@pytest.mark.parametrize("dim", [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reductions_match_torch(name, dim, keepdim):
    k, t = both(np.random.randn(3, 4, 2) * 0.5 + 1)
    if dim is None:
        if keepdim:
            pytest.skip("torch has no keepdim without dim for every reduction")
        expected = getattr(t, name)()
    elif name == "prod":
        if isinstance(dim, tuple):
            pytest.skip("torch.prod takes one dimension")
        expected = t.prod(dim, keepdim=keepdim)
    else:
        expected = getattr(t, name)(dim, keepdim=keepdim)
    assert_same(getattr(k, name)(dim, keepdim), expected)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_max_min_and_arg_match_torch(dim):
    k, t = both(np.random.randn(4, 5))
    assert_same(k.max(dim), t.max(dim).values)
    assert_same(k.min(dim), t.min(dim).values)
    assert k.argmax(dim).tolist() == t.argmax(dim).tolist()
    assert k.argmin(dim).tolist() == t.argmin(dim).tolist()


def test_var_std_norm_match_torch():
    k, t = both(np.random.randn(4, 5))
    assert_same(k.var(1), t.var(1))
    assert_same(k.std(), t.std())
    assert_same(k.var(0, correction=0), t.var(0, correction=0))
    assert_same(k.norm(1), t.norm(dim=1))


@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((4,), (4,)), ((2, 3, 4), (4, 2)), ((4,), (4, 2))])
def test_matmul_matches_torch(shapes):
    ka, ta = both(np.random.randn(*shapes[0]))
    kb, tb = both(np.random.randn(*shapes[1]))
    assert_same(ka @ kb, ta @ tb)
    assert_same(kael.matmul(ka, kb), torch.matmul(ta, tb))


@pytest.mark.parametrize("dim", [0, -1])
def test_softmax_family_matches_torch(dim):
    k, t = both(np.random.randn(4, 5))
    assert_same(kael.softmax(k, dim), torch.softmax(t, dim))
    assert_same(kael.log_softmax(k, dim), torch.log_softmax(t, dim))
    assert_same(kael.logsumexp(k, dim), torch.logsumexp(t, dim))


def test_where_cat_stack_clamp_match_torch():
    ka, ta = both(np.random.randn(3, 4))
    kb, tb = both(np.random.randn(3, 4))
    assert_same(kael.where(ka > 0, ka, kb), torch.where(ta > 0, ta, tb))
    assert_same(kael.where(ka > 0, 1.0, kb), torch.where(ta > 0, torch.tensor(1.0, dtype=torch.float64), tb))
    assert_same(kael.cat([ka, kb], 1), torch.cat([ta, tb], 1))
    assert_same(kael.stack([ka, kb], 1), torch.stack([ta, tb], 1))
    assert_same(ka.clamp(-0.5, 0.5), ta.clamp(-0.5, 0.5))
    assert_same(ka.masked_fill(ka > 0, 7), ta.masked_fill(ta > 0, 7))


def test_indexing_ops_match_torch():
    k, t = both(np.random.randn(4, 5))
    indices = [3, 0, 0]
    assert_same(
        kael.index_select(k, 0, kael.tensor(indices)), t.index_select(0, torch.tensor(indices))
    )
    picks = np.random.randint(0, 5, (4, 2))
    assert_same(kael.gather(k, 1, kael.tensor(picks)), t.gather(1, torch.tensor(picks)))
