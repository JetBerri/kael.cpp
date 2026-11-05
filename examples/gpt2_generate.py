import argparse
import time

import tiktoken

import kael


def tokens_per_second(model, prompt, count, use_cache):
    started = time.perf_counter()
    output = model.generate(prompt, count, temperature=0.0, use_cache=use_cache)
    kael.cuda_synchronize()
    elapsed = time.perf_counter() - started
    return count / elapsed, output


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2 small.")
    parser.add_argument(
        "prompt", nargs="?", default="The meaning of life is", help="text to continue"
    )
    parser.add_argument(
        "--weights",
        default="data/gpt2/model.safetensors",
        help="GPT-2 small safetensors checkpoint",
    )
    parser.add_argument("--tokens", type=int, default=60, help="tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="sample from the K likeliest tokens"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="device to run on"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="report tokens per second with and without the KV cache",
    )
    arguments = parser.parse_args()

    encoding = tiktoken.get_encoding("gpt2")
    model = kael.GPT()
    model.load_gpt2(arguments.weights)
    model.to(arguments.device)
    model.eval()
    prompt = kael.tensor([encoding.encode(arguments.prompt)]).to(arguments.device)

    if arguments.benchmark:
        for use_cache in (True, False):
            rate, _ = tokens_per_second(model, prompt, arguments.tokens, use_cache)
            print(f"use_cache={use_cache}: {rate:.2f} tokens/s")
        return

    kael.manual_seed(arguments.seed)
    output = model.generate(
        prompt,
        arguments.tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
    )
    print(encoding.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
