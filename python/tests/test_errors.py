import os
import pathlib
import subprocess
import sys

import pytest

import kael


def test_index_errors():
    t = kael.zeros(2, 3)
    with pytest.raises(IndexError):
        t.size(5)
    with pytest.raises(IndexError):
        t[9]
    with pytest.raises(IndexError):
        t.select(0, 9)
    with pytest.raises(IndexError):
        kael.index_select(t, 0, kael.tensor([9]))


def test_value_errors():
    t = kael.zeros(2, 3)
    with pytest.raises(ValueError):
        t.view(4)
    with pytest.raises(ValueError):
        t + kael.zeros(4)
    with pytest.raises(ValueError):
        t @ t
    with pytest.raises(ValueError):
        t.item()
    with pytest.raises(ValueError):
        kael.zeros(-1)
    with pytest.raises(ValueError):
        kael.cat([])
    with pytest.raises(ValueError):
        kael.set_num_threads(0)


def test_type_errors():
    with pytest.raises(TypeError):
        kael.neg(kael.zeros(2, dtype=kael.bool))
    with pytest.raises(TypeError):
        kael.mean(kael.zeros(2, dtype=kael.int64))
    with pytest.raises(TypeError):
        kael.where(kael.zeros(2), 1, 2)
    with pytest.raises(TypeError):
        kael.index_select(kael.zeros(2), 0, kael.zeros(1))
    with pytest.raises(TypeError):
        kael.matmul(kael.zeros(2), "x")


def test_runtime_errors():
    with pytest.raises(RuntimeError):
        kael.zeros(2).backward()
    if not kael.cuda_is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            kael.zeros(2).cuda()


def test_messages_name_the_problem():
    with pytest.raises(ValueError, match="broadcastable"):
        kael.zeros(2) + kael.zeros(3)
    with pytest.raises(IndexError, match="out of range"):
        kael.zeros(2)[5]


def test_only_kael_errors_are_translated():
    with pytest.raises(FileNotFoundError):
        pathlib.Path("/nonexistent/file").read_bytes()
    with pytest.raises(RuntimeError):
        from kael import data

        data.load_mnist("/nonexistent/directory")


@pytest.mark.parametrize("configured", ["abc", "0", "-2", "4x", "100000"])
def test_an_invalid_thread_count_in_the_environment_is_rejected(configured):
    script = "import kael; kael.get_num_threads()"
    environment = {**os.environ, "KAEL_NUM_THREADS": configured}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "KAEL_NUM_THREADS must be an integer" in result.stderr
