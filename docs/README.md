# Building kael from scratch

This book explains how kael works and how to rebuild it, one layer at a time. It starts with a flat block of memory and a list of strides, and ends with a GPT-2 checkpoint that loads, tokenizes text and generates it. Each chapter matches one phase of the project, points to the exact files involved and ends with a list of steps you can follow to write that layer yourself.

## Audience

The book targets engineers who write C++ and want to understand what a deep learning framework does internally. It assumes C++20 and basic linear algebra. It does not assume prior knowledge of PyTorch internals, automatic differentiation, CUDA or transformers. Each of those is introduced from first principles in its chapter.

## How to read it

Read the chapters in order. Each one depends on the layers before it, and the code excerpts assume the types introduced earlier.

Every phase is tagged in git. Open a tag in a separate worktree to see the code exactly as it was when that chapter was complete, then read the chapter next to the code:

```bash
git worktree add ../kael-views chapter-04-views
cmake -S ../kael-views -B ../kael-views/build
cmake --build ../kael-views/build
ctest --test-dir ../kael-views/build
```

A separate worktree keeps your main checkout, and this book, on `main`. Early tags carry older drafts of some chapters, so a plain `git checkout` of a tag would replace the book with those drafts.

Each chapter has the same structure:

| Section | Purpose |
|---------|---------|
| Concepts | The idea from first principles, with worked numeric examples and diagrams |
| Implementation | The real types and functions, in the order you would write them |
| Edge cases and pitfalls | Failure modes the code guards against, and how |
| How it is tested | The test suites and golden data that prove the layer correct |
| Build it yourself | Ordered steps to reimplement the layer, each with a checkpoint |

## Chapters

| Chapter | What you build | Tag |
|---------|----------------|-----|
| [00. Overview](00-overview.md) | The architecture, the repository map and the build | |
| [01. Shapes and strides](01-shapes-and-strides.md) | The index arithmetic that turns a flat buffer into an n-dimensional array, and broadcasting | `chapter-02-shapes` |
| [02. Memory model](02-memory-model.md) | Data types, storage, `TensorImpl`, the `Tensor` handle, factories and random numbers | `chapter-03-core` |
| [03. Views](03-views.md) | Reshape, transpose, slicing and expand without copying data | `chapter-04-views` |
| [04. TensorIterator](04-tensor-iterator.md) | The loop engine behind every elementwise op: broadcasting, type promotion and dimension coalescing | `chapter-05-iteration` |
| [05. Operations](05-operations.md) | Arithmetic, math functions, reductions, matmul, indexing and softmax, checked against PyTorch | `chapter-06-ops` |
| [06. Autograd](06-autograd.md) | Reverse-mode automatic differentiation over a dynamic graph | `chapter-07-autograd` |
| [07. Neural networks and training](07-neural-networks.md) | Modules, losses, optimizers, data loading and an MNIST classifier | `chapter-08-training` |
| [08. CPU performance](08-cpu-performance.md) | A thread pool, a fast matmul, a caching allocator and fused kernels | `chapter-09-cpu-performance` |
| [09. CUDA backend](09-cuda-backend.md) | Device memory, device kernels and a host emulation backend for testing | `chapter-10-cuda` |
| [10. Python bindings](10-python-bindings.md) | A Python package on top of the C++ library, with NumPy interop | `chapter-11-python` |
| [11. Transformer](11-transformer.md) | Attention, a GPT model, a KV cache and sampling | `chapter-12-transformer` |
| [12. Loading GPT-2](12-model-loading.md) | The safetensors format, the GPT-2 weight mapping and the BPE tokenizer | `chapter-13-polish` |
| [13. Serving and tooling](13-serving-and-tooling.md) | An HTTP server, debugging tools, fuzzing, sanitizers and CI | `chapter-13-polish` |

Tags are numbered by project phase, so `chapter-02-shapes` holds the code for chapter 01 and `chapter-03-core` the code for chapter 02. Chapter 00 has no tag. Two shape functions arrive after chapter 01: `infer_shape` at `chapter-04-views` and `dim_position` at `chapter-06-ops`. Chapter 12 starts at `chapter-12-transformer` and is complete at `chapter-13-polish`, which also marks `v1.0.0`.

## Design decisions

Architecture decision records explain the choices that shape the whole library:

- [ADR 0001. Runtime data type](adr/0001-runtime-dtype.md)
- [ADR 0002. Tensor as a handle](adr/0002-tensor-as-handle.md)
- [ADR 0003. Type promotion](adr/0003-type-promotion.md)
- [ADR 0004. A PyTorch-shaped API with typed errors](adr/0004-pytorch-shaped-api.md)
- [ADR 0005. Define-by-run autograd graph](adr/0005-dynamic-autograd-graph.md)

## Conventions

- Paths are relative to the repository root. Code excerpts state the file they come from.
- Excerpts are trimmed for reading. The repository is the reference: when an excerpt and the code disagree, the code wins.
- Shapes are written as `[2, 3]`, strides as `(3, 1)` when both appear together, and `numel` is the number of elements.
- The book describes the code at the `main` branch. Earlier tags show earlier, simpler versions of the same files.
