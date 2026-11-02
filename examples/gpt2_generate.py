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
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="The meaning of life is")
    parser.add_argument("--weights", default="data/gpt2/model.safetensors")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark", action="store_true")
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
