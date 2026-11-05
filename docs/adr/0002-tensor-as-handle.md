# ADR 0002: Tensor as a shared handle

## Status

Accepted in phase 3 (tag `chapter-03-core`). Still in force at `v1.0.0`.

## Context

The meaning of `Tensor b = a;` decides how views, autograd and function arguments work. Two models exist:

- Value semantics, as in `std::vector`: a copy duplicates the data.
- Handle semantics, as in PyTorch and NumPy: a copy shares the data.

Value semantics conflicts with the rest of the design:

- Views (`transpose`, `slice`, `expand`, `view`) need several tensors that read and write the same memory.
- Autograd keeps references to forward tensors in graph nodes and accumulates gradients into the leaf tensors that the user holds. A copy would accumulate into the wrong object.
- Passing a tensor by value, or storing it in a container, would copy its whole buffer.

## Decision

`Tensor` holds one `std::shared_ptr<TensorImpl>` and nothing else. Copying a `Tensor` copies the pointer. `TensorImpl` holds the metadata (shape, strides, offset, dtype, device, autograd state) and a `std::shared_ptr<Storage>`, so several impls (views) can share one buffer. `Storage` is not copyable and owns the bytes.

`clone()` is the explicit deep copy. It returns a new contiguous tensor with its own storage.

`Tensor` has no default constructor and a private constructor that validates the impl. `Tensor::from_impl` is the only way to wrap an impl.

## Consequences

- `b = a` does not copy. A write through `b` is visible through `a`. This matches PyTorch and NumPy and surprises readers who expect C++ value semantics.
- `Tensor` follows the rule of zero: the compiler-generated copy, move and destructor are correct.
- Every `Tensor` refers to a validated impl, so member functions never check for an empty handle. Optional tensors use `std::optional<Tensor>`.
- Two identity queries make aliasing testable: `is_same` compares impls, `shares_storage` compares storages.
- Shared mutable state needs guards. The storage carries a version counter that in-place operations increment, and autograd uses it to detect that a tensor saved for the backward pass was modified (chapter 06). In-place operations reject tensors whose elements overlap, such as the output of `expand` (chapter 03).
- `move_to_` changes the device of an existing impl, so every handle to a parameter sees the move. Optimizers that hold references to parameters keep working after `nn::Module::to`.
