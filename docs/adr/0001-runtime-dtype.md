# ADR 0001: Runtime dtype

Status: accepted

## Context

A tensor must be able to hold `float32`, `float64`, `int64` or `bool`. We have to decide whether the element type is part of the C++ type of the tensor (resolved at compile time) or a value the tensor carries with it (resolved at run time).

## Decision

The element type is a run-time value: `enum class DType { F32, F64, I64, Bool }` stored in the tensor metadata. There is a single C++ type, `kael::Tensor`. Code that needs the concrete type (fill, add, convert) gets it through a dispatch function that maps the `DType` to a C++ type and calls a templated lambda with it.

## Alternatives

- **`Tensor<T>` as a template** (the design of the old code). Rejected because:
  - You cannot keep a `std::vector` of tensors with different element types without a type-erasure layer on top.
  - Autograd needs to treat every tensor the same way (graphs mix dtypes).
  - Python bindings need to expose a single `Tensor` type, as PyTorch does.
  - Every operation between different dtypes becomes a combinatorial explosion of templates.

## Consequences

- We need a dispatch mechanism (`kael::dispatch`) and the discipline to use it in every operation.
- Type errors (asking a `float32` tensor for `data_ptr<double>()`) are caught at run time, not at compile time. Clear `KAEL_CHECK` messages make up for it.
- Each kernel is written once as a generic lambda and instantiated for every dtype.
