import argparse
import json
import os
import re
import statistics
import struct
import subprocess
import timeit
from pathlib import Path

import numpy as np
import torch

SIDE = 1000
MATRIX = 1024
OPERATIONS = ["add", "add_broadcast", "sum", "softmax", "matmul"]
KAEL_BENCHMARKS = {
    "add": "BM_CompareAdd",
    "add_broadcast": "BM_CompareAddBroadcast",
    "sum": "BM_CompareSum",
    "softmax": "BM_CompareSoftmax",
    "matmul": "BM_CompareMatmul",
}
LABELS = {
    "add": "add [1000, 1000]",
    "add_broadcast": "add with broadcast",
    "sum": "sum [1000, 1000]",
    "softmax": "softmax [1000, 1000]",
    "matmul": "matmul 1024^3",
}


def best_microseconds(statement, repeat=7, number=5):
    return min(timeit.repeat(statement, repeat=repeat, number=number)) / number * 1e6


def numpy_softmax(values):
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def measure_numpy():
    a = np.random.rand(SIDE, SIDE).astype(np.float32)
    b = np.random.rand(SIDE, SIDE).astype(np.float32)
    row = np.random.rand(SIDE).astype(np.float32)
    left = np.random.rand(MATRIX, MATRIX).astype(np.float32)
    right = np.random.rand(MATRIX, MATRIX).astype(np.float32)
    return {
        "add": best_microseconds(lambda: a + b),
        "add_broadcast": best_microseconds(lambda: a + row),
        "sum": best_microseconds(lambda: a.sum()),
        "softmax": best_microseconds(lambda: numpy_softmax(a)),
        "matmul": best_microseconds(lambda: left @ right),
    }


def measure_torch():
    a = torch.rand(SIDE, SIDE)
    b = torch.rand(SIDE, SIDE)
    row = torch.rand(SIDE)
    left = torch.rand(MATRIX, MATRIX)
    right = torch.rand(MATRIX, MATRIX)
    return {
        "add": best_microseconds(lambda: a + b),
        "add_broadcast": best_microseconds(lambda: a + row),
        "sum": best_microseconds(lambda: a.sum()),
        "softmax": best_microseconds(lambda: torch.softmax(a, -1)),
        "matmul": best_microseconds(lambda: left @ right),
    }


def measure_kael(binary):
    output = subprocess.run(
        [binary, "--benchmark_filter=BM_Compare", "--benchmark_format=json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    times = {
        b["name"].split("/")[0]: b["real_time"]
        for b in json.loads(output)["benchmarks"]
    }
    return {op: times[name] for op, name in KAEL_BENCHMARKS.items()}


def read_idx(path):
    data = Path(path).read_bytes()
    dims = data[3]
    shape = struct.unpack(f">{dims}I", data[4 : 4 + 4 * dims])
    return (
        np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape).copy()
    )


def torch_mnist_epoch_seconds(data_dir, epochs=5):
    images = (
        torch.from_numpy(read_idx(Path(data_dir) / "train-images-idx3-ubyte")).float()
        / 255.0
    )
    images = (images - 0.1307) / 0.3081
    images = images.view(-1, 784)
    labels = torch.from_numpy(
        read_idx(Path(data_dir) / "train-labels-idx1-ubyte")
    ).long()
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    timings = []
    for _ in range(epochs):
        order = torch.randperm(images.shape[0])
        started = timeit.default_timer()
        for first in range(0, images.shape[0], 64):
            batch = order[first : first + 64]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(images[batch]), labels[batch]
            )
            loss.backward()
            optimizer.step()
        timings.append(timeit.default_timer() - started)
    return statistics.median(timings)


def kael_mnist_epoch_seconds(binary, data_dir):
    output = subprocess.run(
        [binary, data_dir], check=True, capture_output=True, text=True
    ).stdout
    return statistics.median(float(t) for t in re.findall(r"in ([0-9.]+) s", output))


def format_time(microseconds):
    if microseconds >= 1000:
        return f"{microseconds / 1000:.2f} ms"
    return f"{microseconds:.0f} us"


def main():
    parser = argparse.ArgumentParser(
        description="Compare kael with NumPy and PyTorch on the same operations."
    )
    parser.add_argument(
        "--bench",
        default="build-release/benchmarks/kael_bench",
        help="kael Google Benchmark binary",
    )
    parser.add_argument(
        "--mnist",
        default="build-release/examples/mnist_mlp",
        help="kael MNIST example binary, timed per epoch",
    )
    parser.add_argument("--data", default="data/mnist", help="MNIST data directory")
    parser.add_argument(
        "--threads",
        type=int,
        default=torch.get_num_threads(),
        help="thread count for kael and PyTorch; NumPy follows its BLAS environment",
    )
    arguments = parser.parse_args()

    os.environ["KAEL_NUM_THREADS"] = str(arguments.threads)
    torch.set_num_threads(arguments.threads)
    kael = measure_kael(arguments.bench)
    numpy_times = measure_numpy()
    torch_times = measure_torch()

    print(
        f"threads: {arguments.threads}  numpy {np.__version__}  torch {torch.__version__}\n"
    )
    print("| Operation | kael | NumPy | PyTorch |")
    print("|---|---|---|---|")
    for operation in OPERATIONS:
        print(
            f"| {LABELS[operation]} | {format_time(kael[operation])} "
            f"| {format_time(numpy_times[operation])} | {format_time(torch_times[operation])} |"
        )
    kael_epoch = kael_mnist_epoch_seconds(arguments.mnist, arguments.data)
    torch_epoch = torch_mnist_epoch_seconds(arguments.data)
    print(f"| MNIST epoch | {kael_epoch:.2f} s | - | {torch_epoch:.2f} s |")


if __name__ == "__main__":
    main()
