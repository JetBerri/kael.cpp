#!/bin/bash
set -uo pipefail

architecture="${1:-native}"
build="${2:-build-cuda}"
report="${3:-cuda-report}"
minimum_cmake="3.24"
minimum_mnist_accuracy="97"
incomplete_exit_code=3

mkdir -p "$report"
rm -f "$report"/*.log
summary="$report/summary.txt"
: >"$summary"
failures=0
skips=0

step() {
    local name="$1"
    shift
    local log="$report/$name.log"
    local started=$SECONDS
    echo "==> $name"
    if "$@" >"$log" 2>&1; then
        printf 'PASS  %-24s %5ss\n' "$name" "$((SECONDS - started))" | tee -a "$summary"
        return 0
    fi
    printf 'FAIL  %-24s %5ss  see %s\n' "$name" "$((SECONDS - started))" "$log" | tee -a "$summary"
    tail -n 25 "$log"
    failures=$((failures + 1))
    return 1
}

skip() {
    printf 'SKIP  %-24s        %s\n' "$1" "$2" | tee -a "$summary"
    skips=$((skips + 1))
}

finish() {
    echo
    echo "--- summary ---"
    cat "$summary"
    if [[ "$failures" -gt 0 ]]; then
        echo "$failures step(s) failed; logs are in $report/"
        exit 1
    fi
    if [[ "$skips" -gt 0 ]]; then
        echo "no step failed, but $skips step(s) did not run: this is not a complete pass"
        exit "$incomplete_exit_code"
    fi
    echo "all steps ran and passed; logs are in $report/"
    exit 0
}

show_environment() {
    uname -a
    cmake --version | head -n 1
    nvcc --version | head -n 2
    "${CXX:-c++}" --version | head -n 1
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    git rev-parse HEAD 2>/dev/null || true
}

check_cmake() {
    local found
    found="$(cmake --version | head -n 1 | awk '{print $3}')"
    [[ "$(printf '%s\n%s\n' "$minimum_cmake" "$found" | sort -V | head -n 1)" == "$minimum_cmake" ]] || {
        echo "CMake $minimum_cmake or newer is required, found $found"
        return 1
    }
}

check_gpu() {
    command -v nvcc || {
        echo "nvcc is not on PATH: install the CUDA Toolkit"
        return 1
    }
    nvidia-smi --query-gpu=name --format=csv,noheader
}

# Without a device the tests skip themselves, and a skipped run must not look like a passed one.
run_device_tests() {
    local output
    output="$(CUDA_LAUNCH_BLOCKING=1 "$build/tests/kael_tests" \
        --gtest_filter='Cuda*:*CudaMatmul*' 2>&1)"
    local status=$?
    echo "$output"
    [[ "$status" -eq 0 ]] || return "$status"
    if echo "$output" | grep -q 'no CUDA device'; then
        echo "device tests were skipped: no CUDA device visible to this process"
        return 1
    fi
    echo "$output" | grep -q '\[  PASSED  \]'
}

run_sanitizer() {
    compute-sanitizer --error-exitcode 1 "$build/tests/kael_tests" \
        --gtest_filter='Cuda.BackwardMatchesCpuAndStaysOnTheDevice:Cuda.IndexingMatchesCpu'
}

# Prints the output so the step log keeps it, and fails below the expected final accuracy.
expect_mnist_accuracy() {
    local output
    output="$("$@" 2>&1)"
    local status=$?
    echo "$output"
    [[ "$status" -eq 0 ]] || return "$status"
    local accuracy
    accuracy="$(echo "$output" | grep 'test accuracy' | tail -n 1 | sed -E 's/.*test accuracy ([0-9.]+)%.*/\1/')"
    [[ -n "$accuracy" ]] || {
        echo "no test accuracy in the output"
        return 1
    }
    awk -v found="$accuracy" -v wanted="$minimum_mnist_accuracy" 'BEGIN { exit !(found + 0 >= wanted + 0) }' || {
        echo "accuracy $accuracy% is below $minimum_mnist_accuracy%"
        return 1
    }
}

run_python_device_tests() {
    local output
    output="$(python_tool -m pytest python/tests/test_cuda.py -q -rs 2>&1)"
    local status=$?
    echo "$output"
    [[ "$status" -eq 0 ]] || return "$status"
    if echo "$output" | grep -q 'skipped'; then
        echo "python device tests were skipped"
        return 1
    fi
}

build_python() {
    local environment="$build/python-venv"
    python3 -m venv "$environment" &&
        "$environment/bin/pip" install --quiet --upgrade pip &&
        "$environment/bin/pip" install --quiet \
            --config-settings=cmake.define.KAEL_ENABLE_CUDA=ON \
            --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="$architecture" . &&
        "$environment/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu &&
        "$environment/bin/pip" install --quiet pytest hypothesis numpy tiktoken transformers \
            safetensors fastapi httpx
}

python_tool() {
    "$build/python-venv/bin/python" "$@"
}

step environment show_environment
step cmake-version check_cmake
step gpu check_gpu

step configure cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DKAEL_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$architecture" -DKAEL_BUILD_EXAMPLES=ON \
    -DKAEL_BUILD_BENCHMARKS=ON || finish
step build cmake --build "$build" --parallel || finish

step ctest ctest --test-dir "$build" --output-on-failure
step device-tests run_device_tests
if command -v compute-sanitizer >/dev/null; then
    step compute-sanitizer run_sanitizer
else
    skip compute-sanitizer "not installed"
fi
step benchmarks "$build/benchmarks/kael_bench" --benchmark_filter='BM_Device'

if [[ -d data/mnist ]]; then
    step mnist-cpu expect_mnist_accuracy "$build/examples/mnist_mlp" data/mnist --device cpu
    step mnist-cuda expect_mnist_accuracy "$build/examples/mnist_mlp" data/mnist --device cuda
    step mnist-cuda-1024 expect_mnist_accuracy "$build/examples/mnist_mlp" data/mnist --device cuda --hidden 1024
else
    skip mnist "data/mnist is missing: run scripts/download_mnist.sh"
fi

if [[ -d data/shakespeare ]]; then
    step char-gpt-cuda "$build/examples/char_gpt" --steps 200 --device cuda
else
    skip char-gpt-cuda "data/shakespeare is missing: run scripts/download_shakespeare.sh"
fi

if [[ -f data/gpt2/vocab.json && -f data/gpt2/merges.txt && -f data/gpt2/model.safetensors ]]; then
    step gpt2-cpp-cuda "$build/examples/gpt2_generate" --tokens 40 --device cuda
else
    skip gpt2-cpp-cuda "GPT-2 files are missing: run scripts/download_gpt2.sh"
fi

if step python-build build_python; then
    step python-sees-gpu python_tool -c 'import kael; assert kael.cuda_is_available()'
    step python-device-tests run_python_device_tests
    step python-tests python_tool -m pytest python/tests -q -rs
    if [[ -d data/mnist ]]; then
        step python-mnist-cuda expect_mnist_accuracy python_tool examples/mnist.py --device cuda
    else
        skip python-mnist-cuda "data/mnist is missing: run scripts/download_mnist.sh"
    fi
    if [[ -f data/gpt2/model.safetensors ]]; then
        step gpt2-cuda python_tool examples/gpt2_generate.py --device cuda
    else
        skip gpt2-cuda "data/gpt2/model.safetensors is missing"
    fi
else
    skip python-steps "the Python package did not build"
fi

finish
