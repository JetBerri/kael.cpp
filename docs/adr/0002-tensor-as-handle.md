# ADR 0002: Tensor as a shared handle

Status: accepted

## Context

We have to decide what copying a `Tensor` means (`Tensor b = a;`): copying the data or sharing it.

## Decision

`Tensor` is a lightweight handle that only holds a `std::shared_ptr<TensorImpl>`. Copying a `Tensor` copies the pointer and bumps the reference count; the data is shared. `TensorImpl` in turn points to a `Storage` through another `shared_ptr`, so several views (different impls) can share the same buffer. `clone()` makes an independent copy.

## Alternatives

- **Value semantics** (copying a tensor copies its data, like `std::vector`). Rejected because:
  - Views (`transpose`, `slice`, `reshape`) need several objects to see the same memory.
  - Autograd keeps references to forward tensors; copying them would be expensive and wrong (gradients must accumulate into the same tensor).
  - Copying megabytes on every assignment or pass-by-value is a huge hidden cost.

## Consequences

- `b = a` does not copy; modifying `b` modifies `a`. This is the PyTorch and NumPy model, but it surprises people who expect C++ value semantics.
- The rule of zero applies: no hand-written copy constructor, assignment or destructor.
- `Tensor` has no default constructor: it always points to a valid impl.
- We need identity queries (`is_same`, `shares_storage`) to reason about aliasing in tests.
