import argparse
import asyncio
import codecs
import json
import resource
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import kael

MAX_PROMPT_CHARACTERS = 20_000
MAX_NEW_TOKENS = 512
LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120)
PEAK_RSS_UNIT = 1 if sys.platform == "darwin" else 1024


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    max_tokens: int = Field(default=64, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.8, ge=0.0, le=10.0)
    top_k: int = Field(default=40, ge=0)
    stream: bool = False


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
                "kael_peak_memory_bytes "
                f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * PEAK_RSS_UNIT}",
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
        if len(ids) >= self.context:
            raise HTTPException(413, f"the prompt has {len(ids)} tokens; the limit is {self.context - 1}")
        return kael.tensor([ids]).to(self.device)

    def run(self, prompt, request, on_token):
        started = time.perf_counter()
        generated = 0

        def receive(token):
            nonlocal generated
            generated += 1
            return on_token(token.item())

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
    app = FastAPI(title="kael GPT server")

    def start_stream(prompt, request):
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
            except Exception as error:
                outcome = error
            loop.call_soon_threadsafe(queue.put_nowait, outcome)

        # The worker owns the slot: a client that never reads the body must not keep it.
        threading.Thread(target=work, daemon=True).start()
        return queue, cancelled

    async def stream_tokens(queue, cancelled):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    generator.metrics.count_request("error")
                    yield sse({"error": "generation failed"})
                    return
                text = decoder.decode(generator.tokenizer.decode([item]))
                if text:
                    yield sse({"text": text})
            tail = decoder.decode(b"", final=True)
            if tail:
                yield sse({"text": tail})
            generator.metrics.count_request("ok")
            yield "data: [DONE]\n\n"
        finally:
            cancelled.set()

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        prompt = generator.prompt_tensor(request)
        generator.reserve()
        if request.stream:
            queue, cancelled = start_stream(prompt, request)
            return StreamingResponse(stream_tokens(queue, cancelled), media_type="text/event-stream")

        tokens = []
        started = time.perf_counter()

        def on_token(token):
            tokens.append(token)
            return True

        try:
            await asyncio.to_thread(generator.run, prompt, request, on_token)
        except Exception as error:
            generator.metrics.count_request("error")
            raise HTTPException(500, "generation failed") from error
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
