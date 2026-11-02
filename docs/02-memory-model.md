# 02. The memory model

This chapter covers phase 3 of kael: the core that turns the shape arithmetic of chapter 01 into real tensors. By the end there is a `Tensor` you can create, read, write, clone, convert and print, in four element types.

## Three layers

A tensor is three objects stacked on top of each other:

```
Tensor              a handle: one shared_ptr, nothing else
  |
  v
TensorImpl          the interpretation: shape, strides, offset, dtype, device
  |
  v
Storage             the memory: a number of bytes on a device
  |
  v
[ raw bytes ... ]
```

Each layer knows only what it needs. `Storage` has no idea what a shape is. `TensorImpl` does not own the bytes, it shares them. `Tensor` holds no data at all, which is why it is cheap to pass around.

The separation pays off in the next chapter: a view is a new `TensorImpl` over the same `Storage`.

## Copying shares, clone copies

`Tensor` follows the rule of zero: it has no hand-written copy constructor, assignment or destructor, because `std::shared_ptr` already does the right thing. Copying a tensor copies the pointer and adds one to a reference count. The data is not touched.

```cpp
kael::Tensor a = kael::zeros({2, 3});
kael::Tensor b = a;          // same tensor
b.at<float>({0, 0}) = 7.0f;
a.at<float>({0, 0});         // 7.0

kael::Tensor c = a.clone();  // independent copy
c.at<float>({0, 0}) = 1.0f;
a.at<float>({0, 0});         // still 7.0
```

Two queries make aliasing testable. `is_same` compares the impl pointers, so it answers "is this literally the same tensor?". `shares_storage` compares the storage pointers, so it answers "can writing through one change the other?". They differ once views exist: two views of one buffer have different impls and the same storage.

There is no default constructor. A `Tensor` always refers to valid data, so no operation has to check for an empty handle. The full reasoning is in [ADR 0002](adr/0002-tensor-as-handle.md).

## The dtype is a run-time value

`DType` is an enum with four values: `F32`, `F64`, `I64`, `Bool`. It lives in the metadata, not in the C++ type, so there is a single `Tensor` type instead of `Tensor<float>`, `Tensor<double>` and so on. That is what lets a `std::vector<Tensor>` hold mixed types, and what autograd and the Python bindings will need. [ADR 0001](adr/0001-runtime-dtype.md) has the alternatives.

The price is that code which touches elements needs the C++ type back. That is the job of `dispatch`, a `switch` that calls a templated lambda with the right type:

```cpp
template <typename F>
decltype(auto) dispatch(DType dtype, F&& f) {
    switch (dtype) {
    case DType::F32:  return f.template operator()<float>();
    case DType::F64:  return f.template operator()<double>();
    case DType::I64:  return f.template operator()<std::int64_t>();
    case DType::Bool: return f.template operator()<std::uint8_t>();
    }
    detail::unreachable();
}
```

An operation is written once, as a lambda, and the compiler instantiates it four times:

```cpp
dispatch(t.dtype(), [&]<typename T>() {
    std::fill_n(t.data_ptr<T>(), t.numel(), convert_value<T>(value));
});
```

Converting a tensor needs the source type and the destination type, so `to()` nests two dispatches. The switches have no `default` on purpose: if a dtype is ever added, `-Wall` flags every switch that does not handle it.

The opposite direction, from a C++ type to a `DType`, is a compile-time trait: `dtype_of_v<float>` is `DType::F32`, and `dtype_of_v<int>` does not compile.

## Why Bool is a uint8_t

Bool elements are stored as one `std::uint8_t` holding 0 or 1, for two reasons:

- `sizeof(bool)` is not fixed by the standard.
- `std::vector<bool>` stores packed bits and cannot hand out a pointer to an element.

So the C++ type behind `DType::Bool` is `std::uint8_t`, and `from_vector` has a separate overload that converts a `std::vector<bool>` element by element.

There is one trap. `static_cast<std::uint8_t>(2.5)` gives 2, which is not a valid bool. `convert_value<T>` handles it: when the target is Bool, any non-zero value becomes 1. That covers `full(shape, 2.5, DType::Bool)` and `to(DType::Bool)`, and NaN becomes true because NaN is not equal to 0.

## Aligned memory

`Storage` allocates with the aligned form of `operator new`:

```cpp
void* raw = ::operator new(nbytes, std::align_val_t{64});
// ...
::operator delete(ptr, std::align_val_t{64});
```

64 bytes is the size of a cache line and what SIMD code will want in a later phase. Every way of allocating has a matching way of freeing, and mixing them is undefined behaviour, so the pointer lives in a `std::unique_ptr` with a custom deleter that can only free it the right way.

Three more rules:

- **Zero bytes means a null pointer.** Allocating 0 bytes has implementation-defined behaviour, so nothing is allocated.
- **The memory is not initialized.** `empty()` really is garbage until written. `zeros` and friends decide what to write.
- **`Storage` is not copyable.** Two owners of one buffer would free it twice. In practice it is always behind a `shared_ptr`.

## Validating metadata

Strides and offset are stored in elements, not bytes. A wrong stride would read somebody else's memory, so every impl is checked when it is wrapped in a `Tensor`. Besides the obvious rules (matching lengths, no negative strides, no negative offset), the check that matters is that the tensor fits inside its storage:

```
max_offset = offset + sum((shape[i] - 1) * strides[i])
needed     = (max_offset + 1) * element_size(dtype)
must hold:   needed <= storage.nbytes()
```

For a contiguous `[2, 3]` float32 tensor with strides `[3, 1]` and offset 0:

```
max_offset = 0 + (2 - 1) * 3 + (3 - 1) * 1 = 5
needed     = (5 + 1) * 4 = 24 bytes
```

For a slice of 10 floats that starts at element 4, with shape `[3]` and stride `[2]`:

```
max_offset = 4 + (3 - 1) * 2 = 8
needed     = (8 + 1) * 4 = 36 bytes, and 36 <= 40
```

If the tensor has no elements nothing is ever read, so there is nothing to check. The multiplications are overflow-checked, so absurd strides throw instead of wrapping around to a small number that passes. This check is the safety net for the next chapter: a view that computes its metadata wrong throws here instead of reading out of bounds.

Kael rejects negative strides outright. NumPy allows them to reverse an axis, but this makes the bound above and all the iteration code more complicated.

## The offset bug nobody sees

`data_ptr<T>()` returns the storage start **plus the offset**. Forgetting the offset is a classic bug that stays invisible until views exist, because until then the offset is always 0. So `test_tensor.cpp` builds an impl by hand with a non-zero offset and checks that `data_ptr` and `at` respect it.

## Reproducible randomness

All random tensors draw from one global `std::mt19937_64`, created on first use inside a function so that it cannot suffer from the static initialization order fiasco. Its default seed is fixed, so a program is reproducible even if it never calls `manual_seed`.

Samples are drawn as `double` and converted, so the same seed gives the same sequence for `F32` and `F64`.

There is a limit worth knowing. The `mt19937_64` engine is fully specified by the standard, but `normal_distribution` and `uniform_real_distribution` are not: libstdc++ on Linux and libc++ on macOS can produce different numbers from the same engine state. The same seed always reproduces on the same machine, not necessarily across platforms. So the tests check properties (same seed gives the same values, the mean of `randn` is close to 0) and never compare against hard-coded numbers. Writing our own distributions would fix this, and it is noted as a TODO. The generator is also not thread-safe yet.

## How this was tested

`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` are on for the library and the tests, and CI turns them into errors. A fourth CI job runs everything under AddressSanitizer and UndefinedBehaviorSanitizer; a temporary test that read `v[10]` of a 3-element vector confirmed that ASan really fires.

Tests are split per component (`test_dtype`, `test_storage`, `test_tensor_impl`, `test_tensor`, `test_factory`, `test_random`). The ones that earn their keep check boundaries: a tensor that needs exactly `nbytes` is valid and one byte less throws, because that is where `<` and `<=` get confused.

## Next chapter

Views: `transpose`, `reshape`, `slice` and `expand`, all without copying a byte.
