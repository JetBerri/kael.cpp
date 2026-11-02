import numpy as np
import pytest

import kael


def test_factories_accept_both_shape_forms():
    assert kael.zeros(2, 3).shape == (2, 3)
    assert kael.zeros((2, 3)).shape == (2, 3)
    assert kael.ones([4]).shape == (4,)
    assert kael.rand(2, 2).shape == (2, 2)
    assert kael.randn((3,)).shape == (3,)
    assert kael.empty(0).numel() == 0


def test_factories_accept_dtype():
    assert kael.zeros(2, dtype=kael.float64).dtype == kael.float64
    assert kael.ones(2, dtype=kael.int64).dtype == kael.int64
    assert kael.full((2,), True, dtype=kael.bool).dtype == kael.bool
    assert kael.zeros(1).dtype == kael.float32


@pytest.mark.parametrize(
    "data, dtype, shape",
    [
        (1, kael.int64, ()),
        (1.5, kael.float32, ()),
        (True, kael.bool, ()),
        ([1, 2, 3], kael.int64, (3,)),
        ([1, 2.5], kael.float32, (2,)),
        ([True, False], kael.bool, (2,)),
        ([[1, 2], [3, 4]], kael.int64, (2, 2)),
        (((1.0,), (2.0,)), kael.float32, (2, 1)),
        ([], kael.float32, (0,)),
    ],
)
def test_tensor_infers_shape_and_dtype(data, dtype, shape):
    t = kael.tensor(data)
    assert t.dtype == dtype
    assert t.shape == shape


def test_tensor_keeps_large_integers_exact():
    big = 2**53 + 1
    assert kael.tensor([big]).item() == big


def test_tensor_rejects_ragged_lists():
    with pytest.raises(ValueError):
        kael.tensor([[1, 2], [3]])
    with pytest.raises(ValueError):
        kael.tensor([[1], 2])


def test_tensor_rejects_foreign_elements():
    with pytest.raises(TypeError):
        kael.tensor(["a"])


def test_tensor_with_explicit_dtype():
    assert kael.tensor([1, 2], dtype=kael.float64).dtype == kael.float64


def test_tensor_keeps_double_precision_when_asked():
    assert kael.tensor(0.1, dtype=kael.float64).item() == 0.1
    assert kael.tensor([[0.1, 0.2]], dtype=kael.float64).tolist() == [[0.1, 0.2]]


def test_tensor_copies_numpy_and_tensors():
    array = np.ones(3, dtype=np.float32)
    t = kael.tensor(array)
    array[0] = 7
    assert t[0].item() == 1
    source = kael.ones(2)
    copy = kael.tensor(source)
    source[0] = 5
    assert copy[0].item() == 1


def test_properties():
    t = kael.zeros(2, 3)
    assert t.ndim == 2
    assert t.numel() == 6
    assert t.size() == (2, 3)
    assert t.size(1) == 3
    assert t.size(-1) == 3
    assert t.stride() == (3, 1)
    assert t.device == "cpu"
    assert t.is_leaf
    assert not t.requires_grad
    assert t.grad is None
    assert t.grad_fn is None


def test_item_returns_python_types():
    assert isinstance(kael.tensor(1.5).item(), float)
    assert isinstance(kael.tensor(2).item(), int)
    assert isinstance(kael.tensor(True).item(), bool)


def test_tolist():
    assert kael.tensor([[1, 2], [3, 4]]).tolist() == [[1, 2], [3, 4]]
    assert kael.tensor(5).tolist() == 5
    assert kael.tensor([True, False]).tolist() == [True, False]


def test_len_and_iter():
    t = kael.arange(4).reshape(2, 2)
    assert len(t) == 2
    assert [row.tolist() for row in t] == [[0, 1], [2, 3]]
    with pytest.raises(TypeError):
        len(kael.tensor(1))


def test_repr_shows_values():
    assert "1.0000" in repr(kael.ones(1))


def test_bool_needs_one_element():
    assert bool(kael.tensor(1.0))
    assert not bool(kael.tensor(0.0))
    with pytest.raises(RuntimeError, match="ambiguous"):
        bool(kael.ones(2))


def test_hash_is_by_identity():
    a = kael.ones(2)
    b = kael.ones(2)
    assert hash(a) != hash(b)
    assert {a: 1}[a] == 1


def test_numeric_conversions():
    assert float(kael.tensor(1.5)) == 1.5
    assert int(kael.tensor(3)) == 3
    assert [10, 20, 30][kael.tensor(1)] == 20


def test_clone_and_detach():
    t = kael.ones(2)
    clone = t.clone()
    clone[0] = 5
    assert t[0].item() == 1
    detached = t.detach()
    detached[0] = 9
    assert t[0].item() == 9


def test_to_dtype_and_device_names():
    t = kael.ones(2)
    assert t.to(kael.float64).dtype == kael.float64
    assert t.to("cpu").shape == (2,)
    assert t.to(dtype=kael.int64).dtype == kael.int64
    like = kael.zeros(1, dtype=kael.float64)
    assert t.to(like).dtype == kael.float64
    with pytest.raises(ValueError):
        t.to("metal")


def test_scalars_mix_on_either_side():
    t = kael.tensor([1.0, 2.0])
    assert (2 * t).tolist() == [2.0, 4.0]
    assert (t * 2).tolist() == [2.0, 4.0]
    assert (1 + t).tolist() == [2.0, 3.0]
    assert (3 - t).tolist() == [2.0, 1.0]
    assert (2 / t).tolist() == [2.0, 1.0]
    assert (2**t).tolist() == [2.0, 4.0]


def test_numpy_scalars_are_numbers():
    t = kael.tensor([1.0, 2.0])
    assert (t * np.float32(2)).tolist() == [2.0, 4.0]
    assert (t + np.int64(1)).tolist() == [2.0, 3.0]


def test_foreign_operands_give_type_errors():
    t = kael.ones(2)
    with pytest.raises(TypeError):
        t + "x"
    with pytest.raises(TypeError):
        t + [1, 2]
    assert (t == "x") is False


def test_in_place_operators_keep_identity():
    t = kael.ones(2)
    original = t
    t += 1
    t *= 3
    t -= 1
    t /= 5
    assert t is original
    assert t.tolist() == [1.0, 1.0]


def test_in_place_methods_return_self():
    t = kael.ones(2)
    assert t.add_(1) is t
    assert t.zero_() is t
    assert t.fill_(2) is t
    assert t.tolist() == [2.0, 2.0]


def test_views_share_memory():
    t = kael.zeros(2, 3)
    t.view(3, 2)[0, 0] = 1
    t.T[0, 1] = 2
    t.transpose(0, 1)[2, 1] = 3
    t.flatten()[1] = 4
    assert t[0, 0].item() == 1
    assert t[1, 0].item() == 2
    assert t[1, 2].item() == 3
    assert t[0, 1].item() == 4


def test_shape_arguments_accept_both_forms():
    t = kael.arange(6)
    assert t.view(2, 3).shape == (2, 3)
    assert t.view((3, 2)).shape == (3, 2)
    assert t.reshape(-1, 2).shape == (3, 2)
    assert kael.zeros(2, 3).permute(1, 0).shape == (3, 2)
    assert kael.zeros(1, 3).expand(2, 3).shape == (2, 3)


def test_split_and_chunk():
    assert [p.shape for p in kael.zeros(5).split(2)] == [(2,), (2,), (1,)]
    assert [p.shape for p in kael.zeros(4).chunk(2)] == [(2,), (2,)]


def test_tensor_rejects_mixed_depth_in_any_order():
    for data in ([1, [2]], [[1], 2], [[1, 2], [3, [4]]], [[[1]], [2]]):
        with pytest.raises(ValueError):
            kael.tensor(data)


def test_tensor_rejects_self_referencing_lists():
    loop = []
    loop.append(loop)
    with pytest.raises(ValueError):
        kael.tensor(loop)


def test_tensor_rejects_absurd_nesting():
    deep = 1.0
    for _ in range(100000):
        deep = [deep]
    with pytest.raises(ValueError):
        kael.tensor(deep)
