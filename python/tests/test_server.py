import asyncio
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import kael

APP_PATH = Path(__file__).resolve().parents[2] / "server" / "app.py"


def load_server():
    spec = importlib.util.spec_from_file_location("kael_server", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load_server()


def byte_symbols():
    printable = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    symbols = {byte: chr(byte) for byte in printable}
    next_symbol = 256
    for byte in range(256):
        if byte not in symbols:
            symbols[byte] = chr(next_symbol)
            next_symbol += 1
    return symbols


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tokenizer")
    vocabulary = {symbol: byte for byte, symbol in byte_symbols().items()}
    (directory / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return kael.BpeTokenizer(str(directory / "vocab.json"), str(directory / "merges.txt"))


@pytest.fixture(scope="module")
def model():
    kael.manual_seed(5)
    tiny = kael.GPT(vocab_size=256, max_positions=64, layers=1, heads=2, dimension=16)
    tiny.eval()
    return tiny


def client_for(model, tokenizer, context=64):
    generator = server.Generator(model, tokenizer, "cpu", context)
    return TestClient(server.create_app(generator)), generator


def wait_until_idle(generator):
    pause = threading.Event()
    for _ in range(100):
        if generator.metrics.in_flight == 0:
            return
        pause.wait(0.05)
    assert generator.metrics.in_flight == 0


def events_of(response):
    return [line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")]


def test_generate_returns_the_requested_token_count(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    response = client.post("/generate", json={"prompt": "hi", "max_tokens": 5})
    assert response.status_code == 200
    assert response.json()["tokens"] == 5


def test_temperature_zero_is_deterministic(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    body = {"prompt": "hi", "max_tokens": 6, "temperature": 0.0}
    assert client.post("/generate", json=body).json()["text"] == client.post("/generate", json=body).json()["text"]


def test_stream_sends_the_same_text_as_a_plain_request(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    body = {"prompt": "hi", "max_tokens": 8, "temperature": 0.0}
    plain = client.post("/generate", json=body).json()["text"]

    response = client.post("/generate", json={**body, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = events_of(response)
    assert events[-1] == "[DONE]"
    assert "".join(json.loads(event)["text"] for event in events[:-1]) == plain


def test_stream_does_not_split_multibyte_characters():
    class Bytes:
        def decode(self, ids):
            return bytes(ids)

    class Sequence:
        def generate(self, prompt, max_tokens, temperature, top_k, on_token):
            for byte in "ñ€".encode():
                on_token(kael.tensor(byte))

    generator = server.Generator(Sequence(), Bytes(), "cpu", 64)
    generator.prompt_tensor = lambda request: None
    client = TestClient(server.create_app(generator))
    events = events_of(client.post("/generate", json={"prompt": "x", "stream": True}))
    assert [json.loads(event)["text"] for event in events[:-1]] == ["ñ", "€"]


def test_only_one_generation_runs_at_a_time(model, tokenizer):
    generator = server.Generator(model, tokenizer, "cpu", 64)
    generator.reserve()
    with pytest.raises(server.HTTPException):
        generator.reserve()
    generator.release()


def test_an_exhausted_server_answers_429_and_recovers(model, tokenizer):
    client, generator = client_for(model, tokenizer)
    generator.reserve()
    response = client.post("/generate", json={"prompt": "hi", "max_tokens": 1})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    generator.release()
    assert client.post("/generate", json={"prompt": "hi", "max_tokens": 1}).status_code == 200


def test_slots_are_released_after_a_stream(model, tokenizer):
    client, generator = client_for(model, tokenizer)
    client.post("/generate", json={"prompt": "hi", "max_tokens": 2, "stream": True})
    wait_until_idle(generator)
    assert client.post("/generate", json={"prompt": "hi", "max_tokens": 1}).status_code == 200


class FakeHttpRequest:
    def __init__(self, body, disconnected=False):
        self.body = body.encode() if isinstance(body, str) else body
        self.disconnected = disconnected

    async def stream(self):
        yield self.body

    async def is_disconnected(self):
        return self.disconnected


def generate_endpoint(generator):
    app = server.create_app(generator)
    return next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/generate")


def test_a_client_that_never_reads_the_stream_does_not_keep_its_slot(model, tokenizer):
    generator = server.Generator(model, tokenizer, "cpu", 64)
    endpoint = generate_endpoint(generator)
    body = json.dumps({"prompt": "hi", "max_tokens": 2, "stream": True})

    async def respond_without_reading():
        await endpoint(FakeHttpRequest(body))
        await asyncio.to_thread(wait_until_idle, generator)

    asyncio.run(respond_without_reading())


class EndlessModel:
    def __init__(self):
        self.tokens = 0

    def generate(self, prompt, max_tokens, temperature, top_k, on_token):
        token = kael.tensor(65)
        while on_token(token):
            self.tokens += 1
            time.sleep(0.01)


def test_a_disconnected_plain_client_cancels_its_generation(tokenizer):
    endless = EndlessModel()
    generator = server.Generator(endless, tokenizer, "cpu", 64)
    endpoint = generate_endpoint(generator)
    body = json.dumps({"prompt": "hi", "max_tokens": 2})

    async def disconnect():
        await endpoint(FakeHttpRequest(body, disconnected=True))
        await asyncio.to_thread(wait_until_idle, generator)

    asyncio.run(disconnect())
    assert generator.metrics.in_flight == 0
    assert 'kael_requests_total{status="cancelled"} 1' in generator.metrics.render()


def test_shutdown_stops_a_running_generation(tokenizer):
    endless = EndlessModel()
    generator = server.Generator(endless, tokenizer, "cpu", 64)
    app = server.create_app(generator)
    with TestClient(app) as client:
        worker = threading.Thread(
            target=client.post,
            args=("/generate",),
            kwargs={"json": {"prompt": "hi", "max_tokens": 1, "stream": True}},
        )
        worker.daemon = True
        worker.start()
        pause = threading.Event()
        for _ in range(500):
            if endless.tokens > 0:
                break
            pause.wait(0.01)
        assert endless.tokens > 0
    assert generator.metrics.in_flight == 0


def test_a_body_larger_than_the_limit_is_rejected_before_parsing(model, tokenizer):
    generator = server.Generator(model, tokenizer, "cpu", 64)
    endpoint = generate_endpoint(generator)
    huge = b"x" * (server.MAX_BODY_BYTES + 1)

    with pytest.raises(server.HTTPException) as caught:
        asyncio.run(endpoint(FakeHttpRequest(huge)))
    assert caught.value.status_code == 413
    assert 'kael_requests_total{status="rejected"} 1' in generator.metrics.render()


def test_the_event_loop_keeps_answering_while_a_prompt_is_encoded(model, tokenizer):
    class SlowTokenizer:
        def encode(self, text):
            time.sleep(0.3)
            return [1]

        def decode(self, ids):
            return b""

    generator = server.Generator(model, SlowTokenizer(), "cpu", 64)
    endpoint = generate_endpoint(generator)
    body = json.dumps({"prompt": "hi", "max_tokens": 1, "stream": True})

    async def scenario():
        request = asyncio.create_task(endpoint(FakeHttpRequest(body)))
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        waited = time.perf_counter() - started
        response = await request
        async for _ in response.body_iterator:
            pass
        return waited

    assert asyncio.run(scenario()) < 0.2


def test_a_failed_generation_answers_500_and_frees_its_slot(tokenizer):
    class Broken:
        def generate(self, *args, **kwargs):
            raise RuntimeError("secret detail")

    generator = server.Generator(Broken(), tokenizer, "cpu", 64)
    client = TestClient(server.create_app(generator), raise_server_exceptions=False)
    response = client.post("/generate", json={"prompt": "hi", "max_tokens": 1})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert generator.metrics.in_flight == 0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "x" * (server.MAX_PROMPT_CHARACTERS + 1)},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "max_tokens": server.MAX_NEW_TOKENS + 1},
        {"prompt": "hi", "temperature": -1},
        {"prompt": "hi", "top_k": -1},
        {"prompt": "hi", "top_k": 2**70},
        {"prompt": "hi", "temperature": 1e-40},
        "not json",
    ],
)
def test_invalid_requests_are_rejected(model, tokenizer, body):
    client, generator = client_for(model, tokenizer)
    assert client.post("/generate", json=body).status_code == 422
    assert generator.metrics.in_flight == 0
    assert 'kael_requests_total{status="rejected"} 1' in generator.metrics.render()


def test_a_lone_surrogate_is_a_rejected_request(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    response = client.post(
        "/generate", content=b'{"prompt": "\\ud800 x"}', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_a_large_top_k_is_clamped_to_the_vocabulary(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    response = client.post("/generate", json={"prompt": "hi", "max_tokens": 1, "top_k": 100000})
    assert response.status_code == 200


def test_a_prompt_longer_than_the_context_is_rejected(model, tokenizer):
    client, generator = client_for(model, tokenizer)
    assert client.post("/generate", json={"prompt": "x" * 64}).status_code == 413
    assert generator.metrics.in_flight == 0


def test_a_request_that_would_outgrow_the_context_is_rejected(model, tokenizer):
    client, generator = client_for(model, tokenizer)
    assert client.post("/generate", json={"prompt": "hi", "max_tokens": 63}).status_code == 413
    assert client.post("/generate", json={"prompt": "hi", "max_tokens": 62}).status_code == 200
    assert generator.metrics.in_flight == 0


def test_health_and_prometheus_metrics(model, tokenizer):
    client, _ = client_for(model, tokenizer)
    assert client.get("/health").json() == {"status": "ok"}
    client.post("/generate", json={"prompt": "hi", "max_tokens": 3})
    text = client.get("/metrics").text
    assert 'kael_requests_total{status="ok"} 1' in text
    assert "kael_generated_tokens_total 3" in text
    assert "kael_requests_in_flight 0" in text
    assert 'kael_request_seconds_bucket{le="+Inf"} 1' in text
    assert "kael_peak_memory_bytes" in text


def test_python_callback_errors_reach_the_caller(model):
    prompt = kael.tensor([[1, 2]])

    def explode(token):
        raise ValueError("stop here")

    with pytest.raises(ValueError, match="stop here"):
        model.generate(prompt, 3, on_token=explode)
    assert model.generate(prompt, 2).shape[1] == 4


@pytest.mark.parametrize("answer", [None, 1, True, "yes"])
def test_truthy_or_missing_callback_answers_continue(model, answer):
    prompt = kael.tensor([[1, 2]])
    calls = []

    def receive(token):
        calls.append(token)
        return answer

    model.generate(prompt, 3, on_token=receive)
    assert len(calls) == 3


@pytest.mark.parametrize("answer", [False, 0, ""])
def test_falsy_callback_answers_stop(model, answer):
    prompt = kael.tensor([[1, 2]])
    calls = []

    def receive(token):
        calls.append(token)
        return answer

    assert model.generate(prompt, 3, on_token=receive).shape[1] == 3
    assert len(calls) == 1


def test_cross_entropy_is_exported_and_matches_the_composed_loss():
    logits = kael.tensor([[1.0, 2.0, 3.0], [0.5, 0.1, -1.0]])
    targets = kael.tensor([2, 0])
    composed = kael.nn.functional.nll_loss(kael.log_softmax(logits, 1), targets)
    assert kael.nn.functional.cross_entropy(logits, targets).item() == pytest.approx(composed.item())
