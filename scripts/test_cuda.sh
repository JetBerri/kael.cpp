#!/bin/bash
set -euo pipefail

architecture="${1:-native}"
build="${2:-build-cuda}"

cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DKAEL_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$architecture" -DKAEL_BUILD_EXAMPLES=ON -DKAEL_BUILD_BENCHMARKS=ON
cmake --build "$build" --parallel

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ctest --test-dir "$build" --output-on-failure

CUDA_LAUNCH_BLOCKING=1 "$build/tests/kael_tests" --gtest_filter='Cuda*:Shapes/CudaMatmul*'
if command -v compute-sanitizer >/dev/null; then
    compute-sanitizer --error-exitcode 1 "$build/tests/kael_tests" \
        --gtest_filter='Cuda.BackwardMatchesCpuAndStaysOnTheDevice:Cuda.IndexingMatchesCpu'
fi

"$build/benchmarks/kael_bench" --benchmark_filter='BM_Device'

if [[ -d data/mnist ]]; then
    "$build/examples/mnist_mlp" data/mnist --device cpu
    "$build/examples/mnist_mlp" data/mnist --device cuda
    "$build/examples/mnist_mlp" data/mnist --device cuda --hidden 1024
fi
