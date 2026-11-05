# kael

A deep learning framework from scratch in C++.

[![CI](https://github.com/JetBerri/kael.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBerri/kael.cpp/actions/workflows/ci.yml)

## What is kael

kael is a deep learning framework written from zero in modern C++. The library depends on nlohmann_json and the system threads library. Tests use GoogleTest, and optional builds add Google Benchmark, BLAS, CUDA with optional cuBLAS, and nanobind. The goal is didactic: understand how a framework like PyTorch works by building one.

## Usage

```cpp
#include <iostream>

#include "kael/kael.h"

int main() {
    kael::Tensor a = kael::arange(0, 6).to(kael::DType::F64);
    kael::Tensor b = a;          // shares data with a
    kael::Tensor c = a.clone();  // independent copy
    b.at<double>({0}) = 7.0;
    std::cout << a << "\n";      // tensor([7.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000], shape=[6], dtype=float64)
}
```

## Milestones

- **v0.1**: train a model on MNIST on CPU, end to end.
- **v0.2**: performance. SIMD, cache-friendly kernels, multithreading.
- **v0.3**: CUDA backend. Tested with host emulation and compile-only builds, not yet on a real GPU.
- **v0.4**: Python bindings.
- **v1.0**: run GPT-2.

## Build and run tests

Requirements: CMake 3.24 or newer and a C++20 compiler. Configuring fetches nlohmann_json and GoogleTest, so it needs network access.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Build the Python package `kael-tensor` and run its tests:

```bash
pip install ".[test]"
pytest python/tests
```

## Repository layout

```text
include/kael/   public headers
src/            implementation
tests/          GoogleTest suites and generated fixtures
python/         Python bindings and pytest suite, package kael-tensor
server/         HTTP generation server and Docker image
examples/       example programs and a quickstart notebook
benchmarks/     performance benchmarks
fuzz/           libFuzzer targets
experiments/    standalone experiments
scripts/        test, fixture generation, data download and GPU scripts
```
