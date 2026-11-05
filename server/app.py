import argparse
import asyncio
import codecs
import json
import resource
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

import kael

MAX_PROMPT_CHARACTERS = 8_192
MAX_NEW_TOKENS = 512
MAX_TOP_K = 1_000_000
MIN_TEMPERATURE = 1e-3
JSON_ESCAPE_FACTOR = 6
MAX_BODY_BYTES = JSON_ESCAPE_FACTOR * MAX_PROMPT_CHARACTERS + 1024
DISCONNECT_POLL_SECONDS = 0.25
SHUTDOWN_WAIT_SECONDS = 30
PENDING = object()
LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120)
PEAK_RSS_UNIT = 1 if sys.platform == "darwin" else 1024


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    max_tokens: int = Field(default=64, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.8, ge=0.0, le=10.0)
    top_k: int = Field(default=40, ge=0, le=MAX_TOP_K)
    stream: bool = False

    @field_validator("temperature")
    @classmethod
    def temperature_is_zero_or_usable(cls, value):
        if 0.0 < value < MIN_TEMPERATURE:
            raise ValueError(f"temperature must be 0 or at least {MIN_TEMPERATURE}")
        return value


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}
        self.tokens = 0
        self.tokens_per_second = 0.0
        self.in_flight = 0
        self.latency_counts = [0] * len(LATENCY_BUCKETS)
        self.latency_sum = 0.0
        self.latency_total = 0

    def count_request(self, status):
        with self.lock:
            self.requests[status] = self.requests.get(status, 0) + 1

    def finish_generation(self, tokens, seconds):
        with self.lock:
            self.tokens += tokens
            self.tokens_per_second = tokens / seconds if seconds > 0 else 0.0
            self.latency_sum += seconds
            self.latency_total += 1
            for index, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    self.latency_counts[index] += 1

    def render(self):
        with self.lock:
            lines = [
                "# TYPE kael_requests_total counter",
                *(
                    f'kael_requests_total{{status="{status}"}} {count}'
                    for status, count in sorted(self.requests.items())
                ),
                "# TYPE kael_generated_tokens_total counter",
                f"kael_generated_tokens_total {self.tokens}",
                "# TYPE kael_tokens_per_second gauge",
                f"kael_tokens_per_second {self.tokens_per_second}",
                "# TYPE kael_requests_in_flight gauge",
                f"kael_requests_in_flight {self.in_flight}",
                "# TYPE kael_request_seconds histogram",
                *(
                    f'kael_request_seconds_bucket{{le="{bound}"}} {count}'
                    for bound, count in zip(LATENCY_BUCKETS, self.latency_counts)
                ),
                f'kael_request_seconds_bucket{{le="+Inf"}} {self.latency_total}',
                f"kael_request_seconds_sum {self.latency_sum}",
                f"kael_request_seconds_count {self.latency_total}",
                "# TYPE kael_peak_memory_bytes gauge",
                (
                    "kael_peak_memory_bytes "
                    f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * PEAK_RSS_UNIT}"
                ),
            ]
        return "\n".join(lines) + "\n"


class Generator:
    """Owns the model, which only one generation can use at a time."""

    def __init__(self, model, tokenizer, device, context):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context = context
        self.slot = threading.Lock()
        self.shutting_down = threading.Event()
        self.metrics = Metrics()

    def reserve(self):
        if not self.slot.acquire(blocking=False):
            self.metrics.count_request("busy")
            raise HTTPException(429, "the model is busy", headers={"Retry-After": "1"})
        with self.metrics.lock:
            self.metrics.in_flight += 1

    def release(self):
        with self.metrics.lock:
            self.metrics.in_flight -= 1
        self.slot.release()

    def prompt_tensor(self, request):
        ids = self.tokenizer.encode(request.prompt)
        if not ids:
            raise HTTPException(422, "the prompt has no tokens")
        if len(ids) + request.max_tokens > self.context:
            raise HTTPException(
                413,
                f"{len(ids)} prompt tokens and {request.max_tokens} new tokens "
                f"do not fit a context of {self.context}",
            )
        return kael.tensor([ids]).to(self.device)

    def run(self, prompt, request, on_token):
        started = time.perf_counter()
        generated = 0

        def receive(token):
            nonlocal generated
            generated += 1
            return on_token(token.item()) and not self.shutting_down.is_set()

        try:
            self.model.generate(
                prompt,
                request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                on_token=receive,
            )
        finally:
            self.metrics.finish_generation(generated, time.perf_counter() - started)
            self.release()


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def create_app(generator):
    @asynccontextmanager
    async def lifespan(_):
        yield
        generator.shutting_down.set()
        if await asyncio.to_thread(generator.slot.acquire, True, SHUTDOWN_WAIT_SECONDS):
            generator.slot.release()

    app = FastAPI(title="kael GPT server", lifespan=lifespan)

    async def parse_request(http_request):
        chunks = []
        size = 0
        async for chunk in http_request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                generator.metrics.count_request("rejected")
                raise HTTPException(
                    413, f"the request is larger than {MAX_BODY_BYTES} bytes"
                )
            chunks.append(chunk)
        try:
            return GenerateRequest.model_validate_json(b"".join(chunks))
        except ValidationError as error:
            generator.metrics.count_request("rejected")
            details = error.errors(
                include_url=False, include_input=False, include_context=False
            )
            raise HTTPException(422, details) from error

    async def prepare_prompt(request):
        generator.reserve()
        try:
            return await asyncio.to_thread(generator.prompt_tensor, request)
        except HTTPException:
            generator.metrics.count_request("rejected")
            generator.release()
            raise
        except Exception as error:
            generator.metrics.count_request("error")
            generator.release()
            raise HTTPException(500, "generation failed") from error
        except asyncio.CancelledError:
            generator.metrics.count_request("cancelled")
            generator.release()
            raise

    def start_generation(prompt, request):
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        cancelled = threading.Event()

        def on_token(token):
            loop.call_soon_threadsafe(queue.put_nowait, token)
            return not cancelled.is_set()

        def work():
            try:
                generator.run(prompt, request, on_token)
                outcome = None
            except Exception as error:  # noqa: BLE001
                outcome = error
            loop.call_soon_threadsafe(queue.put_nowait, outcome)

        # The worker owns the slot: a client that never reads the body must not keep it.
        threading.Thread(target=work, daemon=True).start()
        return queue, cancelled

    async def stream_tokens(queue, cancelled):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        finished = False
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    finished = True
                    generator.metrics.count_request("error")
                    yield sse({"error": "generation failed"})
                    return
                text = decoder.decode(generator.tokenizer.decode([item]))
                if text:
                    yield sse({"text": text})
            tail = decoder.decode(b"", final=True)
            if tail:
                yield sse({"text": tail})
            finished = True
            generator.metrics.count_request("ok")
            yield "data: [DONE]\n\n"
        finally:
            cancelled.set()
            if not finished:
                generator.metrics.count_request("cancelled")

    async def collect_tokens(queue, cancelled, http_request):
        tokens = []
        next_check = time.monotonic() + DISCONNECT_POLL_SECONDS
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), DISCONNECT_POLL_SECONDS)
            except asyncio.TimeoutError:
                item = PENDING
            if item is None:
                return tokens
            if isinstance(item, Exception):
                generator.metrics.count_request("error")
                raise HTTPException(500, "generation failed") from item
            if item is not PENDING:
                tokens.append(item)
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + DISCONNECT_POLL_SECONDS
                if await http_request.is_disconnected():
                    cancelled.set()
                    generator.metrics.count_request("cancelled")
                    return None

    @app.post("/generate")
    async def generate(http_request: Request):
        request = await parse_request(http_request)
        prompt = await prepare_prompt(request)
        started = time.perf_counter()
        queue, cancelled = start_generation(prompt, request)
        if request.stream:
            return StreamingResponse(
                stream_tokens(queue, cancelled), media_type="text/event-stream"
            )

        tokens = await collect_tokens(queue, cancelled, http_request)
        if tokens is None:
            return Response(status_code=499)
        generator.metrics.count_request("ok")
        return {
            "text": generator.tokenizer.decode(tokens).decode("utf-8", "replace"),
            "tokens": len(tokens),
            "seconds": time.perf_counter() - started,
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return generator.metrics.render()

    return app


def load_app(data, device):
    data = Path(data)
    tokenizer = kael.BpeTokenizer(str(data / "vocab.json"), str(data / "merges.txt"))
    model = kael.GPT()
    model.load_gpt2(str(data / "model.safetensors"))
    model.to(device)
    model.eval()
    return create_app(Generator(model, tokenizer, device, context=1024))


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    app = load_app(arguments.data, arguments.device)
    uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
