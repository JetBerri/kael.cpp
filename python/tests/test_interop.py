import gc
import threading
import weakref

import numpy as np
import pytest

import kael


@pytest.mark.parametrize(
    "numpy_dtype, dtype",
    [
        (np.float32, kael.float32),
        (np.float64, kael.float64),
        (np.int64, kael.int64),
        (np.bool_, kael.bool),
    ],
)
def test_from_numpy_shares_memory(numpy_dtype, dtype):
    array = np.zeros((2, 3), dtype=numpy_dtype)
    t = kael.from_numpy(array)
    assert t.dtype == dtype
    assert t.shape == (2, 3)
    array[1, 2] = 1
    assert t[1, 2].item() == 1
    t[0, 0] = 1
    assert array[0, 0] == 1


def test_from_numpy_respects_strides():
    array = np.arange(24.0).reshape(4, 6)
    view = array[1::2, ::3]
    t = kael.from_numpy(view)
    assert t.tolist() == view.tolist()
    assert not t.is_contiguous()
    t[0, 0] = -1
    assert array[1, 0] == -1


def test_from_numpy_on_fortran_order():
    array = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    assert kael.from_numpy(array).tolist() == array.tolist()


def test_from_numpy_zero_dimensional_and_empty():
    assert kael.from_numpy(np.array(2.5)).item() == 2.5
    assert kael.from_numpy(np.zeros((0, 3))).shape == (0, 3)


def test_from_numpy_rejects_what_kael_cannot_represent():
    with pytest.raises(TypeError):
        kael.from_numpy(np.zeros(2, dtype=np.int32))
    with pytest.raises(TypeError):
        kael.from_numpy(np.zeros(2, dtype=np.complex64))
    with pytest.raises(TypeError):
        kael.from_numpy(np.zeros(2, dtype=">f8"))
    with pytest.raises(ValueError):
        kael.from_numpy(np.arange(4.0)[::-1])


def test_from_numpy_copies_read_only_arrays():
    array = np.ones(2)
    array.flags.writeable = False
    t = kael.from_numpy(array)
    t[0] = 5
    assert array[0] == 1


def test_from_numpy_keeps_the_array_alive():
    array = np.arange(1000.0)
    reference = weakref.ref(array)
    t = kael.from_numpy(array)
    del array
    gc.collect()
    assert reference() is not None
    assert t[999].item() == 999
    del t
    gc.collect()
    assert reference() is None


def test_view_of_from_numpy_keeps_the_array_alive():
    array = np.arange(1000.0)
    reference = weakref.ref(array)
    tail = kael.from_numpy(array)[500:]
    del array
    gc.collect()
    assert reference() is not None
    assert tail[0].item() == 500
    del tail
    gc.collect()
    assert reference() is None


def test_numpy_shares_memory():
    t = kael.zeros(2, 3)
    array = t.numpy()
    array[1, 2] = 7
    assert t[1, 2].item() == 7
    t[0, 0] = 3
    assert array[0, 0] == 3


@pytest.mark.parametrize(
    "dtype, numpy_dtype",
    [
        (kael.float32, np.float32),
        (kael.float64, np.float64),
        (kael.int64, np.int64),
        (kael.bool, np.bool_),
    ],
)
def test_numpy_dtypes(dtype, numpy_dtype):
    assert kael.zeros(2, dtype=dtype).numpy().dtype == numpy_dtype


def test_numpy_of_strided_tensor():
    t = kael.arange(12).reshape(3, 4)
    assert t.T.numpy().tolist() == t.T.tolist()
    assert t[:, ::2].numpy().tolist() == t[:, ::2].tolist()


def test_numpy_keeps_the_tensor_memory_alive():
    t = kael.arange(100000, dtype=kael.float64)
    array = t.numpy()
    del t
    gc.collect()
    kael.empty_cache()
    other = kael.zeros(100000, dtype=kael.float64)
    assert array[99999] == 99999
    assert other[0].item() == 0


def test_numpy_of_view_keeps_memory_alive():
    array = kael.arange(1000, dtype=kael.float64)[10:20].numpy()
    gc.collect()
    kael.empty_cache()
    assert array[0] == 10


def test_numpy_refuses_tensors_with_grad():
    with pytest.raises(RuntimeError, match="detach"):
        kael.ones(2).requires_grad_().numpy()
    assert kael.ones(2).requires_grad_().detach().numpy().tolist() == [1.0, 1.0]


def test_np_asarray_and_dtype_argument():
    t = kael.tensor([1.0, 2.0])
    assert np.asarray(t).tolist() == [1.0, 2.0]
    assert np.asarray(t, dtype=np.float64).dtype == np.float64
    copied = np.array(t, copy=True)
    copied[0] = 9
    assert t[0].item() == 1


def test_numpy_functions_accept_tensors():
    assert np.sum(np.asarray(kael.ones(3))) == 3


def test_round_trip_through_numpy_needs_no_copy():
    t = kael.ones(3, dtype=kael.float64)
    again = kael.from_numpy(t.numpy())
    again[0] = 5
    assert t[0].item() == 5


def test_dlpack_with_torch_shares_memory():
    torch = pytest.importorskip("torch")
    t = kael.arange(6, dtype=kael.float32).reshape(2, 3)
    shared = torch.from_dlpack(t)
    assert shared.tolist() == t.tolist()
    shared[0, 0] = 42
    assert t[0, 0].item() == 42


def test_dlpack_keeps_memory_alive():
    torch = pytest.importorskip("torch")
    t = kael.arange(100000, dtype=kael.float64)
    shared = torch.from_dlpack(t)
    del t
    gc.collect()
    kael.empty_cache()
    assert shared[99999].item() == 99999


def test_dlpack_device():
    assert kael.ones(1).__dlpack_device__() == (1, 0)


def test_gil_is_released_during_matmul():
    a = kael.rand(1500, 1500)
    b = kael.rand(1500, 1500)
    done = threading.Event()
    ticks = [0]

    def work():
        for _ in range(3):
            kael.matmul(a, b)
        done.set()

    worker = threading.Thread(target=work)
    worker.start()
    while not done.is_set():
        ticks[0] += 1
    worker.join()
    assert ticks[0] > 100


def test_gil_is_released_during_backward():
    x = kael.rand(800, 800).requires_grad_()
    y = (x @ x @ x).sum()
    done = threading.Event()
    ticks = [0]

    def work():
        y.backward()
        done.set()

    worker = threading.Thread(target=work)
    worker.start()
    while not done.is_set():
        ticks[0] += 1
    worker.join()
    assert x.grad is not None
    assert ticks[0] > 10
