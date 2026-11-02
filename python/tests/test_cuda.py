import numpy as np
import pytest

import kael
from kael import nn, optim
from kael.nn import functional as F

pytestmark = pytest.mark.skipif(not kael.cuda_is_available(), reason="no CUDA device")


def test_factories_and_moves():
    t = kael.zeros(2, 3, device="cuda")
    assert t.device == "cuda"
    assert t.cuda() is not None
    assert t.cpu().device == "cpu"
    assert kael.rand(2, device="cuda").device == "cuda"
    assert kael.arange(3, device="cuda").tolist() == [0, 1, 2]
    assert kael.ones(2).to("cuda").device == "cuda"


def test_values_cross_the_device_boundary():
    t = kael.tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert t[1, 1].item() == 4.0
    assert "device=cuda" in repr(t)
    t[0, 0] = 9
    assert t[0, 0].item() == 9


def test_host_views_are_rejected():
    with pytest.raises(RuntimeError, match="cpu"):
        kael.ones(2).cuda().numpy()
    with pytest.raises(RuntimeError):
        kael.ones(2).cuda().__dlpack_device__()


def test_mixed_devices_are_rejected():
    with pytest.raises(RuntimeError, match="same device"):
        kael.ones(2).cuda() + kael.ones(2)


def test_bool_mask_and_tensor_index_on_device():
    t = kael.tensor([1.0, 2.0, 3.0]).cuda()
    assert t[t > 1].tolist() == [2.0, 3.0]
    assert t[kael.tensor([2, 0]).cuda()].tolist() == [3.0, 1.0]


def test_module_to_and_training_match_cpu():
    def run(device):
        kael.manual_seed(9)
        model = nn.Sequential(nn.Linear(4, 8, dtype=kael.float64), nn.ReLU(), nn.Linear(8, 3, dtype=kael.float64)).to(device)
        optimizer = optim.AdamW(model.parameters())
        x = kael.randn(16, 4, dtype=kael.float64).to(device)
        y = kael.randint(0, 3, (16,)).to(device)
        for _ in range(10):
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
        return [p.detach().cpu().numpy().copy() for p in model.parameters()]

    for on_cpu, on_device in zip(run("cpu"), run("cuda")):
        np.testing.assert_allclose(on_device, on_cpu, rtol=1e-8, atol=1e-8)


def test_dropout_on_device():
    assert F.dropout(kael.ones(8, device="cuda"), 0.5).device == "cuda"
