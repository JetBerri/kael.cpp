# ADR 0004: A PyTorch-shaped API with typed errors

## Status

Accepted in phase 3 (tag `chapter-03-core`) and extended in every later phase. Still in force at `v1.0.0`.

## Context

kael is didactic: a reader should be able to put kael and PyTorch side by side and recognize every call. The API also has to work from C++, where there are no keyword arguments, and from Python, where errors must surface as ordinary Python exceptions.

The alternatives were a naming scheme of kael's own, which would force readers to translate every example, and error codes or `std::expected` results, which would make every arithmetic expression unreadable and give Python callers nothing idiomatic to catch.

## Decision

- Names and semantics follow PyTorch: `sum`, `matmul`, `view`, `reshape`, `transpose`, `masked_fill`, `scatter_add_`, `nn::Linear`, `optim::Adam`, `no_grad`, `backward`. A trailing underscore marks an in-place operation.
- Operations exist as free functions in `kael::` (`include/kael/ops.h`) and, where PyTorch has them, as `Tensor` methods. Arithmetic and comparisons are operator overloads.
- Numbers mix with tensors through `Scalar` and `Operand` (`include/kael/scalar.h`, `include/kael/operand.h`), so `t * 2`, `2 * t` and `t * other` are all valid.
- Functions whose result must be used are `[[nodiscard]]`.
- Errors are exceptions. `kael::Error` derives from `std::runtime_error`, and `IndexError`, `ValueError` and `TypeError` derive from it. The `KAEL_CHECK`, `KAEL_CHECK_INDEX`, `KAEL_CHECK_VALUE` and `KAEL_CHECK_TYPE` macros in `include/kael/error.h` throw them with a message that names the operation and the offending values.
- The Python module registers a translator in `python/src/bindings.cpp` that maps `IndexError`, `ValueError` and `TypeError` to the Python exceptions of the same name and any other `kael::Error` to `RuntimeError`.

## Consequences

- PyTorch code ports to kael almost line by line, and the golden tests can call the same operation on both sides.
- Error messages are part of the behavior: tests match on them, and the book quotes them.
- Exceptions cost nothing on the success path and keep expressions such as `(a * b).sum()` readable, at the price of making failure paths implicit in C++ code.
- Where kael deliberately differs from PyTorch, such as Bool + Bool or `reshape` copying every non-contiguous input, the difference is documented in the chapter that introduces the operation.
