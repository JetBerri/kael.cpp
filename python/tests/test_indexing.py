import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import kael


def base(shape=(3, 4, 5)):
    return np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)


KEYS = [
    0,
    -1,
    (1, 2),
    (1, -1, 3),
    slice(None),
    slice(1, 3),
    slice(0, 3, 2),
    slice(None, None, 2),
    slice(-2, None),
    slice(5, 9),
    None,
    (None, 0),
    (0, None),
    (0, None, 1),
    Ellipsis,
    (Ellipsis, 0),
    (0, Ellipsis),
    (0, Ellipsis, 1),
    (Ellipsis, None),
    (slice(1, 2), slice(None), 3),
    (1, slice(None), None),
    (None, Ellipsis, None),
    (slice(None), slice(1, 3), slice(0, 5, 2)),
]


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_getitem_matches_numpy(key):
    array = base()
    expected = array[key]
    actual = kael.tensor(array)[key]
    assert actual.shape == expected.shape
    assert actual.tolist() == expected.tolist()


def test_guide_example_shape():
    assert kael.zeros(3, 4, 5)[1, :, None].shape == (4, 1, 5)


def test_getitem_is_a_view():
    t = kael.zeros(3, 4)
    t[1:, 2:][0, 0] = 5
    assert t[1, 2].item() == 5


@pytest.mark.parametrize(
    "key",
    [(0, 0, 0, 0), 3, -4, (0, 4), (Ellipsis, Ellipsis)],
    ids=repr,
)
def test_invalid_keys_raise_index_error(key):
    with pytest.raises(IndexError):
        kael.zeros(3, 4, 5)[key]


def test_unsupported_key_types_raise():
    with pytest.raises(IndexError):
        kael.zeros(2)["x"]
    with pytest.raises(IndexError):
        kael.zeros(2)[1.5]


def test_negative_step_is_rejected():
    with pytest.raises(ValueError):
        kael.zeros(3)[::-1]


def test_integer_tensor_key_matches_numpy():
    array = base()
    indices = [2, 0, 0]
    assert kael.tensor(array)[kael.tensor(indices)].tolist() == array[indices].tolist()
    assert (
        kael.tensor(array)[:, kael.tensor([1, 3])].tolist() == array[:, [1, 3]].tolist()
    )
    assert kael.tensor(array)[kael.tensor(1)].tolist() == array[1].tolist()


def test_bool_mask_key_matches_numpy():
    array = base()
    mask = array % 2 == 0
    assert kael.tensor(array)[kael.tensor(mask)].tolist() == array[mask].tolist()
    with pytest.raises(IndexError):
        kael.zeros(3)[kael.tensor([True])]


def test_float_tensor_key_is_a_type_error():
    with pytest.raises(TypeError):
        kael.zeros(3)[kael.zeros(1)]


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 7),
        (slice(1, 2), 7),
        ((0, 1), 7.0),
        ((Ellipsis, 0), 3),
        ((slice(None), 1), np.int64(4)),
    ],
    ids=repr,
)
def test_setitem_scalar_matches_numpy(key, value):
    array = base().astype(np.float32)
    t = kael.tensor(array)
    array[key] = value
    t[key] = value
    assert t.tolist() == array.tolist()


def test_setitem_tensor_broadcasts_like_numpy():
    array = base().astype(np.float32)
    t = kael.tensor(array)
    update = np.ones((4, 5), dtype=np.float32) * 9
    array[1] = update
    t[1] = kael.tensor(update)
    assert t.tolist() == array.tolist()
    array[:, 0] = np.arange(5, dtype=np.float32)
    t[:, 0] = kael.arange(5, dtype=kael.float32)
    assert t.tolist() == array.tolist()


def test_setitem_with_bool_mask():
    t = kael.tensor([1.0, 2.0, 3.0])
    t[t > 1] = 0
    assert t.tolist() == [1.0, 0.0, 0.0]


def test_setitem_with_int_tensor_is_rejected():
    with pytest.raises(TypeError):
        kael.zeros(3)[kael.tensor([0])] = 1


def _key_items(size):
    return st.one_of(
        st.integers(-size, size - 1),
        st.builds(
            slice,
            st.one_of(st.none(), st.integers(-size - 2, size + 2)),
            st.one_of(st.none(), st.integers(-size - 2, size + 2)),
            st.one_of(st.none(), st.integers(1, 3)),
        ),
    )


@st.composite
def shapes_and_keys(draw):
    shape = tuple(draw(st.lists(st.integers(1, 4), min_size=1, max_size=4)))
    used = draw(st.integers(0, len(shape)))
    items = [draw(_key_items(shape[i])) for i in range(used)]
    position = draw(st.integers(0, len(items)))
    extra = draw(st.sampled_from([None, Ellipsis, "plain"]))
    if extra != "plain":
        items.insert(position, extra)
    return shape, tuple(items)


@given(shapes_and_keys())
@settings(max_examples=300, deadline=None)
def test_random_keys_match_numpy(case):
    shape, key = case
    array = base(shape)
    try:
        expected = array[key]
    except IndexError:
        with pytest.raises(IndexError):
            kael.tensor(array)[key]
        return
    actual = kael.tensor(array)[key]
    assert actual.shape == expected.shape
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize(
    "key",
    [
        lambda: (0, kael.tensor([1])),
        lambda: (kael.tensor([1]),),
        lambda: (kael.tensor([0]), 0),
        lambda: (slice(None), kael.tensor([1])),
    ],
)
def test_assigning_through_a_copying_key_is_rejected(key):
    t = kael.zeros(3, 3)
    with pytest.raises(TypeError):
        t[key()] = 1
    assert t.sum().item() == 0


def test_assigning_through_a_scalar_tensor_key_writes():
    t = kael.zeros(3)
    t[kael.tensor(1)] = 5
    assert t.tolist() == [0.0, 5.0, 0.0]


def test_mask_in_a_one_element_tuple_assigns():
    t = kael.tensor([1.0, 2.0])
    t[(t > 1,)] = 5
    assert t.tolist() == [1.0, 5.0]


@pytest.mark.parametrize("key", [True, False, (0, True)], ids=repr)
def test_bool_keys_are_rejected(key):
    t = kael.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(IndexError):
        t[key]
    with pytest.raises(IndexError):
        t[key] = 9
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]
