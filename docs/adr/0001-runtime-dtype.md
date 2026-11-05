# ADR 0001: Runtime dtype

## Status

Accepted in phase 3 (tag `chapter-03-core`). Still in force at `v1.0.0`.

## Context

A tensor holds `float32`, `float64`, `int64` or `bool` elements. The element type can be part of the C++ type of the tensor, resolved at compile time (`Tensor<float>`), or a value the tensor carries, resolved at run time.

A templated `Tensor<T>` gives compile-time type checks and no dispatch cost, but it conflicts with the rest of the design:

- A container of tensors with different element types needs a type-erasure layer on top of `Tensor<T>`.
- The autograd graph stores inputs and gradients of every dtype in the same nodes.
- The Python bindings expose one `Tensor` type, as PyTorch does.
- Operations between two dtypes need a template for every pair, and type promotion becomes a compile-time computation over every combination.

## Decision

The element type is a runtime value, `enum class DType { F32, F64, I64, Bool }` in `include/kael/dtype.h`, stored in `TensorImpl::dtype`. There is one C++ class, `kael::Tensor`.

Code that needs the concrete type calls `dispatch(dtype, f)`, a `switch` that calls the templated lambda `f` with `float`, `double`, `std::int64_t` or `std::uint8_t`. `dispatch_floating` and `dispatch_no_bool` restrict the set and throw `TypeError` for the other dtypes. The traits `dtype_of_v<T>` and `cpp_type_t<D>` map between the two worlds at compile time.

`Bool` is stored as `std::uint8_t` holding 0 or 1, because `sizeof(bool)` is implementation defined and `std::vector<bool>` cannot hand out element pointers. `convert_value` maps any non-zero value to 1 when the target is `Bool`.

## Consequences

- Every operation that touches elements goes through `dispatch` or one of its variants. A kernel is written once, as a generic lambda, and instantiated for each dtype it supports.
- Dtype mismatches are runtime errors. `data_ptr<double>()` on a `float32` tensor throws `TypeError` with `expected dtype float64 but tensor has float32`, instead of failing to compile.
- Type promotion is a runtime function, `promote_types`, with the order `Bool < I64 < F32 < F64`. The iterator applies it once per operation (chapter 04).
- `switch` statements over `DType` have no `default` branch, so `-Wall` flags every switch that misses a new enumerator if a dtype is added.
- Each instantiated kernel adds code for every dtype. The restricted dispatchers keep float-only kernels out of the integer instantiations.
