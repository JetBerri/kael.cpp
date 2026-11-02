import math
from pathlib import Path

import numpy as np
import pytest

import kael

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "gpt2" / "model.safetensors"


def assert_close(actual, expected, tolerance):
    np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy(), atol=tolerance, rtol=tolerance)


def tiny_hf_model():
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=50, n_positions=32, n_embd=24, n_layer=3, n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0
    )
    return transformers.GPT2LMHeadModel(config).eval()


def save_for_kael(model, path):
    state = {name: tensor.clone().contiguous() for name, tensor in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    safetensors_torch.save_file(state, str(path))


def tiny_kael_model(path):
    model = kael.GPT(vocab_size=50, max_positions=32, layers=3, heads=4, dimension=24)
    model.load_gpt2(str(path))
    model.eval()
    return model


def test_logits_match_a_small_huggingface_model(tmp_path):
    reference = tiny_hf_model()
    path = tmp_path / "tiny.safetensors"
    save_for_kael(reference, path)
    model = tiny_kael_model(path)
    tokens = np.random.randint(0, 50, (2, 12)).astype(np.int64)
    with torch.no_grad():
        expected = reference(torch.tensor(tokens)).logits
    actual = model(kael.tensor(tokens))
    np.testing.assert_allclose(np.array(actual.tolist()), expected.numpy(), atol=2e-5, rtol=2e-4)


def test_greedy_generation_matches_huggingface(tmp_path):
    reference = tiny_hf_model()
    path = tmp_path / "tiny.safetensors"
    save_for_kael(reference, path)
    model = tiny_kael_model(path)
    prompt = np.array([[1, 7, 3]], dtype=np.int64)
    with torch.no_grad():
        expected = reference.generate(
            torch.tensor(prompt), max_new_tokens=10, do_sample=False, pad_token_id=0
        )
    for use_cache in (True, False):
        actual = model.generate(kael.tensor(prompt), 10, temperature=0.0, use_cache=use_cache)
        assert actual.tolist() == expected.tolist()


def test_attention_matches_torch_scaled_dot_product():
    x = np.random.randn(1, 5, 8).astype(np.float64)
    q = kael.tensor(x)
    t = torch.tensor(x)
    mask = kael.triu(kael.ones(5, 5, dtype=kael.bool), 1)
    scores = kael.matmul(q, q.mT) / math.sqrt(8)
    output = kael.matmul(kael.softmax(kael.masked_fill(scores, mask, -math.inf), -1), q)
    expected = torch.nn.functional.scaled_dot_product_attention(t, t, t, is_causal=True)
    np.testing.assert_allclose(np.array(output.tolist()), expected.numpy(), atol=1e-10)


def test_safetensors_written_by_python_load_in_kael(tmp_path):
    tensors = {
        "a": torch.randn(3, 2),
        "b": torch.randn(2, dtype=torch.float64),
        "c": torch.tensor([1, 2, 3]),
        "d": torch.tensor([True, False]),
    }
    path = tmp_path / "python.safetensors"
    safetensors_torch.save_file(tensors, str(path), metadata={"format": "pt"})
    loaded = kael.load_safetensors(str(path))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].tolist() == tensor.tolist()
    assert kael.read_safetensors_metadata(str(path)) == {"format": "pt"}


def test_safetensors_written_by_kael_load_in_python(tmp_path):
    tensors = {
        "a": kael.randn(3, 2),
        "b": kael.randn(2, dtype=kael.float64),
        "c": kael.tensor([1, 2, 3]),
        "d": kael.tensor([True, False]),
        "e": kael.zeros(0, 2),
    }
    path = tmp_path / "kael.safetensors"
    kael.save_safetensors(str(path), tensors, {"format": "kael"})
    loaded = safetensors_torch.load_file(str(path))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].tolist() == tensor.tolist()


def test_sequence_ops_match_torch():
    array = np.random.randn(4, 6)
    k = kael.tensor(array, dtype=kael.float64)
    t = torch.tensor(array)
    for dim in (0, 1, -1):
        assert_close(torch.tensor(kael.cumsum(k, dim).tolist(), dtype=torch.float64), torch.cumsum(t, dim), 1e-12)
        values, indices = kael.sort(k, dim)
        expected = torch.sort(t, dim)
        assert indices.tolist() == expected.indices.tolist()
        assert values.tolist() == expected.values.tolist()
        values, indices = kael.sort(k, dim, descending=True)
        assert indices.tolist() == torch.sort(t, dim, descending=True).indices.tolist()
    for largest in (True, False):
        values, indices = kael.topk(k, 3, 1, largest)
        expected = torch.topk(t, 3, 1, largest)
        assert values.tolist() == expected.values.tolist()
        assert indices.tolist() == expected.indices.tolist()
    for diagonal in (-1, 0, 2):
        assert kael.tril(k, diagonal).tolist() == torch.tril(t, diagonal).tolist()
        assert kael.triu(k, diagonal).tolist() == torch.triu(t, diagonal).tolist()


def test_model_from_python_trains_with_python_optimizer():
    kael.manual_seed(0)
    model = kael.GPT(vocab_size=10, max_positions=8, layers=1, heads=2, dimension=8)
    optimizer = kael.optim.AdamW(model.parameters(), lr=3e-3)
    tokens = kael.arange(9).unsqueeze(0)
    inputs, targets = tokens[:, :8], tokens[:, 1:].reshape(8)
    for _ in range(200):
        loss = kael.nn.functional.cross_entropy(model(inputs).view(8, 10), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.2


@pytest.mark.skipif(not WEIGHTS.exists(), reason="GPT-2 weights are not downloaded")
def test_real_gpt2_logits_match_huggingface():
    tiktoken = pytest.importorskip("tiktoken")
    tokens = tiktoken.get_encoding("gpt2").encode("Hello, my name is")
    model = kael.GPT()
    model.load_gpt2(str(WEIGHTS))
    model.eval()
    actual = model(kael.tensor([tokens]))
    assert model.num_parameters() == 124439808

    config = transformers.GPT2Config.from_pretrained(str(WEIGHTS.parent))
    reference = transformers.GPT2LMHeadModel(config).eval()
    state = safetensors_torch.load_file(str(WEIGHTS))
    state = {name.removeprefix("transformer."): tensor for name, tensor in state.items()}
    reference.transformer.load_state_dict(state, strict=False)
    reference.tie_weights()
    with torch.no_grad():
        expected = reference(torch.tensor([tokens])).logits
    difference = np.abs(actual[0, -1].detach().numpy() - expected[0, -1].numpy()).max()
    print("max logit difference", difference)
    assert difference < 1e-3, difference
