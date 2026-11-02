import numpy as np
import pytest

import kael

torch = pytest.importorskip("torch")


def pair(array):
    k = kael.tensor(array, dtype=kael.float64).requires_grad_()
    t = torch.tensor(array, dtype=torch.float64, requires_grad=True)
    return k, t


def assert_grads_match(k, t):
    np.testing.assert_allclose(np.array(k.grad.tolist()), t.grad.numpy(), rtol=1e-9, atol=1e-10)


CASES = {
    "arithmetic": lambda x, lib: (x * x + 2 * x - x / 3).sum(),
    "exp_tanh": lambda x, lib: (lib.exp(x) * lib.tanh(x)).sum(),
    "matmul": lambda x, lib: (x @ x.mT).sum(),
    "mean": lambda x, lib: (x**2).mean(),
    "softmax": lambda x, lib: (lib.softmax(x, -1) * x).sum(),
    "log_softmax": lambda x, lib: lib.log_softmax(x, 1).sum(),
    "views": lambda x, lib: (x.T[1:] * 2).sum() + x.reshape(-1).sum() + x[0, 1:].sum(),
    "stack": lambda x, lib: lib.stack([x, x * 2]).sum(),
    "cat": lambda x, lib: (lib.cat([x, x], 0) ** 2).sum(),
    "where": lambda x, lib: lib.where(x > 0, x, x * 3).sum(),
    "clamp": lambda x, lib: x.clamp(-0.5, 0.5).sum(),
}


@pytest.mark.parametrize("name", CASES)
def test_gradients_match_torch(name):
    array = np.random.randn(3, 4)
    k, t = pair(array)
    CASES[name](k, kael).backward()
    CASES[name](t, torch).backward()
    assert_grads_match(k, t)


def test_non_scalar_backward_with_gradient():
    k, t = pair(np.random.randn(3))
    weights = [1.0, 2.0, 3.0]
    (k * 2).backward(kael.tensor(weights, dtype=kael.float64))
    (t * 2).backward(torch.tensor(weights, dtype=torch.float64))
    assert_grads_match(k, t)


def test_scalar_needed_without_gradient():
    with pytest.raises(ValueError):
        (kael.ones(2).requires_grad_() * 2).backward()


def test_retain_graph_flag():
    x = kael.ones(2).requires_grad_()
    y = (x * x).sum()
    y.backward(retain_graph=True)
    y.backward()
    assert x.grad.tolist() == [4.0, 4.0]
    with pytest.raises(RuntimeError, match="second time"):
        y.backward()


def test_grad_fn_and_leaf():
    x = kael.ones(2).requires_grad_()
    y = x * 2
    assert x.is_leaf and not y.is_leaf
    assert y.grad_fn is not None and "Mul" in y.grad_fn
    assert x.grad_fn is None


def test_no_grad_context_manager():
    x = kael.ones(2).requires_grad_()
    with kael.no_grad():
        assert not kael.is_grad_enabled()
        y = x * 2
    assert kael.is_grad_enabled()
    assert not y.requires_grad


def test_no_grad_restores_after_exception():
    with pytest.raises(ZeroDivisionError):
        with kael.no_grad():
            1 / 0
    assert kael.is_grad_enabled()


def test_no_grad_decorator():
    @kael.no_grad()
    def double(x):
        return x * 2

    x = kael.ones(2).requires_grad_()
    assert not double(x).requires_grad
    assert double.__name__ == "double"
    assert kael.is_grad_enabled()


def test_enable_grad_inside_no_grad():
    x = kael.ones(2).requires_grad_()
    with kael.no_grad():
        with kael.enable_grad():
            assert (x * 2).requires_grad
        assert not (x * 2).requires_grad


def test_no_grad_object_can_nest_with_itself():
    guard = kael.no_grad()
    with guard:
        with guard:
            pass
        assert not kael.is_grad_enabled()
    assert kael.is_grad_enabled()


def test_requires_grad_is_writable():
    x = kael.ones(2)
    x.requires_grad = True
    assert x.requires_grad
    with pytest.raises(RuntimeError):
        (x * 2).requires_grad = False
    with pytest.raises(RuntimeError):
        kael.zeros(1, dtype=kael.int64).requires_grad_()


def test_gradcheck_from_python():
    x = kael.randn(3, dtype=kael.float64).requires_grad_()
    passed, report = kael.gradcheck(lambda inputs: kael.tanh(inputs[0]) * inputs[0], [x])
    assert passed, report


def test_gradcheck_callback_can_use_python_after_backward():
    calls = []

    def function(inputs):
        calls.append(1)
        return (inputs[0] * inputs[0]).sum()

    x = kael.randn(2, dtype=kael.float64).requires_grad_()
    passed, report = kael.gradcheck(function, [x])
    assert passed, report
    assert calls


def test_in_place_on_leaf_is_rejected():
    x = kael.ones(2).requires_grad_()
    with pytest.raises(RuntimeError):
        x += 1


def test_modifying_saved_tensor_is_detected():
    x = kael.ones(2).requires_grad_()
    y = x * 1
    z = y * y
    with kael.no_grad():
        y.add_(1)
    with pytest.raises(RuntimeError, match="inplace"):
        z.sum().backward()
