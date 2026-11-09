# ADR 0003: Type promotion

## Status

Accepted in phase 5 (tag `chapter-05-iteration`) with `promote_types` and `promote_with_scalar`. The 0-d rule (`result_dtype`) and the per-operation overrides (`add_dtype`, `div_dtype`, `check_no_bool`, `check_result_fits`) arrived in phase 6 (tag `chapter-06-ops`). Still in force at `v1.0.0`.

## Context

A binary operation can mix dtypes (`int64 + float32`) and Python or C++ numbers (`x * 0.5`). The result dtype needs a rule, and there are three common families:

- Strict typing, as in JAX without promotion: mixed inputs are an error and the caller casts. It is explicit, but every loss scaled by a literal needs a cast.
- Value-based promotion, as in NumPy before 2.0: the dtype of a scalar depends on its value. The result type then changes with the data, which makes kernels and tests harder to reason about.
- Category-based promotion, as in PyTorch: dtypes are ranked, and a scalar or 0-d tensor only matters when it belongs to a higher category (Bool, integer, floating).

The main use of the library is reproducing PyTorch results, and its tests compare against PyTorch golden data, so the rule must match PyTorch wherever possible.

## Decision

kael follows PyTorch's category-based rules:

- `promote_types` in `include/kael/dtype.h` ranks `Bool < I64 < F32 < F64` and returns the higher dtype.
- `dtype_category` groups the dtypes into Bool (0), integer (1) and floating (2).
- A `Scalar` takes its dtype from its C++ type, not its value (`2` is I64, `2.5` is F32). `promote_with_scalar` lets it change the result only when its category is higher than the tensor's.
- `result_dtype` in `include/kael/kernels.h` promotes tensors with dimensions and 0-d tensors separately. The 0-d group wins only with a higher category.
- A few ops override the default: `add` turns Bool + Bool into I64, `div` turns integer results into F32, `sub`, `pow` and `floor_divide` reject Bool, and in-place ops refuse a result whose category is higher than `self`.

The kernel layer computes the result dtype once per operation and converts the inputs before they reach the iterator: `binary_kernel` calls `result_dtype`, and `same_dtype_kernel` adds each input as `input.to(dtype)`. `TensorIterator::build` rejects inputs of different dtypes, so kernels never see mixed dtypes.

## Consequences

- `x * 0.5` on a float32 tensor stays float32, and a float64 0-d loss does not upcast a large float32 activation.
- `int64 + float32` gives float32, as in PyTorch, so integers above 2^24 lose precision.
- Bool + Bool gives int64, unlike PyTorch's Bool. The golden generator skips the Bool pattern for `add` (`BOOL_PATTERNS[1:]` in `scripts/generate/golden.py`).
- The rules are plain `constexpr` functions, checked at compile time by the `static_assert`s in `Promotion.Table` and `Promotion.Scalars` in `tests/test_iterator.cpp`. The run-time behavior of the operations is tested in `tests/test_ops_binary.cpp`.
