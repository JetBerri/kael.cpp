# kael

A deep learning framework from scratch in C++. 🚧 Work in progress.

[![CI](https://github.com/JetBerri/kael.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBerri/kael.cpp/actions/workflows/ci.yml)

## What is kael

kael is a deep learning framework written from zero in modern C++, with no dependencies beyond the standard library and GoogleTest. The goal is didactic: understand how a framework like PyTorch works by building one. Each phase ships three things together: the code, the tests that prove it, and a chapter that explains it.

## Usage

```cpp
#include "kael/kael.h"

kael::Tensor a = kael::arange(0, 6).to(kael::DType::F64);
kael::Tensor b = a;          // shares data with a
kael::Tensor c = a.clone();  // independent copy
b.at<double>({0}) = 7.0;
std::cout << a << "\n";      // tensor([7.0000, 1.0000, ...], shape=[6], dtype=float64)
```

## Roadmap

- **v0.1**: train a model on MNIST on CPU, end to end.
- **v0.2**: performance. SIMD, cache-friendly kernels, multithreading.
- **v0.3**: CUDA backend.
- **v0.4**: Python bindings.
- **v1.0**: run GPT-2.

## Build and run tests

Requirements: CMake 3.24 or newer and a C++20 compiler. GoogleTest is fetched automatically.

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Repository layout

```
include/kael/   public headers
src/            implementation
tests/          GoogleTest suites and generated fixtures
scripts/        helper scripts (test runner, NumPy fixture generator)
docs/           one chapter per phase
benchmarks/     performance benchmarks (reserved for v0.2)
examples/       example programs (reserved)
python/         Python bindings (reserved for v0.4)
legacy/         old experiments kept for reference
```

## Documentation

All chapters live in [docs/](docs/).

- [01. Shapes and strides](docs/01-shapes-and-strides.md)
- [02. The memory model](docs/02-memory-model.md)
