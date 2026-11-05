# 01. Shapes and strides

This chapter builds the metadata arithmetic that turns a flat block of memory into an n-dimensional tensor. It maps to phase 2 of the project. There is no `Tensor` class yet: everything here is a set of free functions over two vector types, and every later layer (views, the iterator, broadcasting in operations, autograd's gradient reduction) calls them.

| File | Role |
|------|------|
| `include/kael/shape.h` | `Shape` and `Strides` type aliases and the declarations of every function in this chapter |
| `src/shape.cpp` | Implementation: validation, element count, contiguous strides, offsets, broadcasting, dimension wrapping, shape inference |
| `src/internal/checked_math.h` | `detail::checked_mul` and `detail::checked_add`, overflow-checked arithmetic on `std::int64_t` |
| `include/kael/error.h` | `kael::Error`, its subclasses and the `KAEL_CHECK_*` macros used to report bad input |
| `scripts/gen_broadcast_cases.py` | Generates random broadcasting cases with NumPy |
| `tests/generated/broadcast_cases.inc` | The generated cases, included by the test |
| `tests/test_shape.cpp` | GoogleTest suites for every function |

Two functions in `shape.h` arrived in later phases but belong to the same family: `wrap_dim` came with the `Tensor` handle in phase 3, `infer_shape` came with `view` and `reshape` in phase 4, and `dim_position` came with a refactor in phase 6. This chapter covers them so that `shape.h` is complete.

## Prerequisites

- [00. Overview](00-overview.md) for the build commands and the layout of the repository.
- Familiarity with C++ `std::vector`, `std::span` and exceptions. No tensor knowledge is assumed.

## A tensor is a buffer plus metadata

In memory, a tensor is a flat array of numbers plus a few values that say how to read it:

- **Shape**: the number of elements along each dimension. A matrix with 2 rows and 3 columns has shape `[2, 3]`.
- **Strides**: for each dimension, how many elements to skip in the flat array to move one step along that dimension.
- **Offset**: the position of the first element inside the buffer.

```text
buffer:  [ 1.0  2.0  3.0  4.0  5.0  6.0 ]

shape:   [2, 3]
strides: [3, 1]
offset:  0
```

The buffer stores the data. The metadata stores the interpretation. Keeping the two apart is what lets later chapters transpose, slice and broadcast a tensor without copying any data: those operations only produce new metadata.

kael represents shapes and strides as plain vectors of signed 64-bit integers.

From `include/kael/shape.h`:

```cpp
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;
```

The two aliases name the same type, so the compiler does not stop you from passing strides where a shape is expected. The aliases exist for the reader. A strong type per concept would catch that mix-up, but it would also force conversions at every call site that does arithmetic between the two, and the tests catch the mistake anyway.

The element type is `std::int64_t` for three reasons:

- Products of dimensions grow fast. A `[50000, 50000]` float tensor has 2.5 billion elements, which overflows a 32-bit `int`.
- Strides, offsets and indices take part in the same arithmetic. One signed width for all of them avoids mixed-width conversions, which `-Wconversion -Wsign-conversion` would flag on every line.
- Signed values let the validation code detect a negative dimension instead of seeing a huge unsigned number.

## Shape and element count

The shape is the list of dimension sizes. The element count, `numel`, is their product.

```text
shape        numel
[2, 3, 4]    24
[5]          5
[]           1
[3, 0, 2]    0
```

Two rows deserve attention.

The shape `[]` has no dimensions. The product of an empty list is 1, so it holds exactly one element. This is a **0-d tensor**, also called a scalar tensor. It is not a special case in the code: the loop in `numel` runs zero times and returns its initial value.

The shape `[3, 0, 2]` has a dimension of size 0, so it holds no elements. Empty tensors are legal. A batch with no samples or a slice with `start == end` produces one, and NumPy and PyTorch both allow them. Only negative dimensions are an error.

### Validation

`validate_shape` rejects negative dimensions and nothing else.

From `src/shape.cpp`:

```cpp
void validate_shape(const Shape& shape) {
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK_VALUE(shape[i] >= 0,
                         "negative dimension: " << shape[i] << " at position " << i
                                                << " in shape: " << to_string(shape));
    }
}
```

`KAEL_CHECK_VALUE` throws `kael::ValueError` with the file, the line, the failed condition and the streamed message. Chapter 02 describes the error macros in detail. For a shape `[2, -2]`, the message reads:

```text
kael error at <path>/src/shape.cpp:<line>: check 'shape[i] >= 0' failed: negative dimension: -2 at position 1 in shape: [2, -2]
```

`to_string(const Shape&)` formats a shape as `[2, 3]` and `[]` for a 0-d shape. Every error message in kael uses it, so shapes always print the same way.

### The overflow check

`numel` multiplies the dimensions, and the product can exceed the range of `std::int64_t`. Signed overflow in C++ is undefined behavior, so the code cannot multiply first and check the result afterwards: the compiler may assume the overflow never happens and remove the check.

The check must run before each multiplication. If `total * dim` would exceed the maximum `max`, then `total > max / dim`, and that division cannot overflow.

From `src/shape.cpp`:

```cpp
std::int64_t numel(const Shape& shape) {
    validate_shape(shape);

    if (std::ranges::find(shape, 0) != shape.end()) {
        return 0;
    }

    std::int64_t total = 1;
    for (const std::int64_t dim : shape) {
        KAEL_CHECK_VALUE(total <= std::numeric_limits<std::int64_t>::max() / dim,
                         "number of elements overflows int64 for shape " << to_string(shape));
        total *= dim;
    }
    return total;
}
```

The early return for a zero dimension does two jobs. It gives the right answer (any product with a 0 is 0), and it guarantees that `dim` is at least 1 inside the loop, so `max / dim` never divides by zero.

The order also matters for correctness. Without the early return, `numel({1LL << 40, 1LL << 40, 0})` would throw on the first two dimensions even though the tensor has no elements.

A worked boundary, from the `Numel.Overflow` test:

```text
numel({max})         = max                (fits exactly)
numel({max, 2})      -> throws            (max <= max / 2 is false)
numel({2^31, 2^31})  = 2^62               (fits)
numel({2^40, 2^40})  -> throws            (2^80 does not fit)
```

### Checked arithmetic helpers

The same pattern repeats in several places, so `src/internal/checked_math.h` packages it for non-negative operands.

From `src/internal/checked_math.h`:

```cpp
inline std::int64_t checked_mul(std::int64_t a, std::int64_t b, const char* quantity) {
    KAEL_CHECK(a >= 0 && b >= 0, "checked_mul expects non-negative operands for " << quantity);
    if (a == 0 || b == 0) {
        return 0;
    }
    KAEL_CHECK_VALUE(a <= std::numeric_limits<std::int64_t>::max() / b,
                     quantity << " overflows int64 (" << a << " * " << b << ")");
    return a * b;
}
```

`checked_add` follows the same shape with `a <= max - b`. The `quantity` argument names what is being computed (`"tensor stride"`, `"tensor size in bytes"`), so the error message tells you which calculation overflowed. The header is private: it lives under `src/`, not `include/`, because it is an implementation detail and not part of the public API.

## Row-major layout and strides

A matrix of 2 rows and 3 columns is stored one row after the other. This is **row-major order**, the default in C, NumPy and PyTorch.

```text
logical view (shape [2, 3]):

        col 0   col 1   col 2
row 0 [  a       b       c   ]
row 1 [  d       e       f   ]

flat memory:

position:  0    1    2    3    4    5
value:   [ a    b    c    d    e    f ]
```

To move one column to the right, skip 1 element. To move one row down, skip 3 elements, a whole row. Those distances are the strides: `[3, 1]`.

### Computing contiguous strides

The strides of a row-major (contiguous) layout come from a running product, built from the last dimension to the first:

```text
shape [2, 3, 4]

start with stride = 1
dim 2 (size 4): strides[2] = 1    then stride = 1 * 4  = 4
dim 1 (size 3): strides[1] = 4    then stride = 4 * 3  = 12
dim 0 (size 2): strides[0] = 12   then stride = 12 * 2 = 24

strides = [12, 4, 1]
```

The last dimension always has stride 1 because its elements are adjacent. Each dimension to its left skips everything to its right.

From `src/shape.cpp`:

```cpp
Strides contiguous_strides(const Shape& shape) {
    validate_shape(shape);

    Strides strides(shape.size());
    std::int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride = detail::checked_mul(stride, std::max<std::int64_t>(shape[i], 1), "tensor stride");
    }
    return strides;
}
```

The loop header `for (std::size_t i = shape.size(); i-- > 0;)` is the idiom for counting down with an unsigned index. The test `i-- > 0` compares before decrementing, so the body sees `size - 1` down to `0` and the loop ends without wrapping around.

### Why `max(dim, 1)`

The running product multiplies by `max(dim, 1)` instead of `dim`. A dimension of size 0 has no elements, so its stride is never used to read memory. But a plain product would set every stride to its left to 0, and the tensor would lose the layout of its other dimensions:

```text
shape [2, 0, 3]  ->  strides [3, 3, 1]   (with max(dim, 1))
shape [2, 0, 3]  ->  strides [0, 0, 1]   (with a plain product)
```

Well-formed strides matter in chapter 03, where views of an empty tensor go through the same stride arithmetic as views of a full one, and in the contiguity check, which compares strides against this function. PyTorch produces the same strides: `torch.empty(2, 0, 3).stride()` is `(3, 3, 1)`. NumPy differs here and reports all-zero strides for empty arrays.

The product is overflow-checked even though `numel` would catch most overflows, because `max(dim, 1)` changes the product. A shape like `[2^40, 0, 2^40]` has 0 elements, so `numel` returns 0, but its contiguous strides would need `2^80`. `contiguous_strides` throws `ValueError` with `tensor stride overflows int64` for it, and so does `zeros` with that shape.

## From index to offset

Given a multidimensional index, the position of the element in the flat buffer is:

```text
offset = base_offset + sum over i of index[i] * strides[i]
```

Worked example. Shape `[2, 3, 4]`, strides `[12, 4, 1]`, base offset 0, index `[1, 0, 1]`:

```text
dimension 0:  1 * 12 = 12
dimension 1:  0 *  4 =  0
dimension 2:  1 *  1 =  1
                     -----
offset             =  13
```

The same formula works for any strides, not only contiguous ones. That is the property the rest of the library relies on: views change the strides and the base offset, and element access keeps using this formula.

From `src/shape.cpp`:

```cpp
std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                               std::span<const std::int64_t> strides, std::int64_t base_offset,
                               std::span<const std::int64_t> index) {
    // two KAEL_CHECK_VALUE calls: shape, strides and index have the same size
    std::int64_t offset = base_offset;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        KAEL_CHECK_INDEX(index[i] >= 0 && index[i] < shape[i],
                         "index " << index[i] << " out of range [0, " << shape[i]
                                  << ") at dimension " << i << " in shape " << join(shape));
        offset += index[i] * strides[i];
    }
    return offset;
}
```

Three design points:

- The parameters are `std::span<const std::int64_t>`, not `const Shape&`. A span accepts a `Shape`, a `std::array`, or a pointer and a length without copying, so callers that keep indices in other containers do not have to build a `Shape` first.
- An index outside `[0, shape[i])` throws `kael::IndexError`, not `ValueError`. The Python bindings map `IndexError` to Python's `IndexError`, so `t[5]` on a 3-element tensor raises the exception a Python user expects.
- The function accepts only non-negative indices. Python-style negative indices (`-1` for the last element) are resolved one layer up, in `Tensor::at`, which adds the dimension size to each negative index before calling this function. Keeping this layer strict means a computed index that went negative by mistake fails here instead of silently reading another element.

The products `index[i] * strides[i]` are not overflow-checked. The index is in range and the strides belong to a tensor whose extent was already validated against its storage (chapter 02), so every partial sum is bounded by the size of the buffer.

For a 0-d shape, all three spans are empty, the loop does nothing, and the result is the base offset. That is the position of the single element.

## From linear position to index

The inverse problem: given a position `linear` in row-major order, recover the multidimensional index. Walk the shape from right to left. The remainder of the division by the dimension size is the coordinate in that dimension. The quotient is what remains for the dimensions to the left.

Linear position 13 in shape `[2, 3, 4]`:

```text
dimension 2 (size 4):  13 % 4 = 1    remaining 13 / 4 = 3
dimension 1 (size 3):   3 % 3 = 0    remaining  3 / 3 = 1
dimension 0 (size 2):   1 % 2 = 1    remaining  1 / 2 = 0

index = [1, 0, 1]
```

That is the index from the previous section, which confirms that the two functions are inverses for contiguous strides.

From `src/shape.cpp`:

```cpp
Shape index_from_linear(std::int64_t linear, const Shape& shape) {
    const std::int64_t count = numel(shape);
    KAEL_CHECK_INDEX(linear >= 0 && linear < count, "linear index "
                                                        << linear << " out of range [0, " << count
                                                        << ") for shape " << to_string(shape));

    Shape index(shape.size());
    for (std::size_t i = shape.size(); i-- > 0;) {
        index[i] = linear % shape[i];
        linear /= shape[i];
    }
    return index;
}
```

The range check comes first and also protects the division. An empty shape has `count == 0`, so every `linear` fails the check and the loop never divides by a zero dimension. For a 0-d shape, `count == 1`, `linear` must be 0, and the result is the empty index `[]`.

`index_from_linear` is the slow, general way to visit every element: one division per dimension per element. The tensor printer in chapter 02 and several tests use it. The iterator in chapter 04 replaces it with incremental counters for the hot loops.

## Transposing without copying

Keeping data and metadata apart pays off here. Take the same six values:

```text
buffer: [ a  b  c  d  e  f ]
```

Read with shape `[2, 3]` and strides `[3, 1]`:

```text
[ a  b  c ]
[ d  e  f ]
```

Read with shape `[3, 2]` and strides `[1, 3]`:

```text
[ a  d ]
[ b  e ]
[ c  f ]
```

The second reading is the transpose, and no value moved. Check with the offset formula: element `[2, 1]` of the transposed reading is at `2 * 1 + 1 * 3 = 5`, which holds `f`. In the original, `f` is at `[1, 2]`, which is `[2, 1]` with the coordinates swapped.

Swapping two entries of the shape and the same two entries of the strides is the whole transpose. Chapter 03 builds `transpose`, `permute`, `slice` and the other views on this idea.

## Broadcasting shapes

Broadcasting lets an operation combine tensors of different shapes, for example adding a `[3]` bias to every row of a `[2, 3]` matrix, without first copying the bias into two rows. It is a rule that combines two shapes into one result shape.

The rules, identical to NumPy's:

1. Align the shapes from the right. If one shape has fewer dimensions, treat the missing leading dimensions as size 1.
2. Two sizes are compatible if they are equal or if one of them is 1.
3. The result size is the larger of the two, or the common value when they are equal.

If any pair fails rule 2, the shapes are not broadcastable.

```text
a:       [5, 1, 4]
b:          [3, 1]      missing leading dimension counts as 1

aligned:
a:        5   1   4
b:        1   3   1
          |   |   |
result:   5   3   4
```

Size 0 follows the same rules, because the rule says "one of them is 1", not "the smaller one". `[0]` with `[1]` gives `[0]`, but `[0]` with `[3]` fails. NumPy behaves the same way.

| a | b | result |
|---|---|--------|
| `[3]` | `[2, 3]` | `[2, 3]` |
| `[2, 1]` | `[1, 3]` | `[2, 3]` |
| `[]` | `[4]` | `[4]` |
| `[]` | `[]` | `[]` |
| `[0]` | `[1]` | `[0]` |
| `[0]` | `[3]` | error |
| `[2, 3]` | `[3, 2]` | error |

From `src/shape.cpp`:

```cpp
Shape broadcast_shapes(const Shape& a, const Shape& b) {
    validate_shape(a);
    validate_shape(b);

    const std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);
    for (std::size_t from_right = 0; from_right < ndim; ++from_right) {
        const std::int64_t dim_a = from_right < a.size() ? a[a.size() - 1 - from_right] : 1;
        const std::int64_t dim_b = from_right < b.size() ? b[b.size() - 1 - from_right] : 1;
        KAEL_CHECK_VALUE(dim_a == dim_b || dim_a == 1 || dim_b == 1,
                         "shapes " << to_string(a) << " and " << to_string(b)
                                   << " are not broadcastable");
        result[ndim - 1 - from_right] = dim_a == 1 ? dim_b : dim_a;
    }
    return result;
}
```

The loop counts positions from the right (`from_right`), which handles the alignment and the implicit leading 1s in one place. The result line `dim_a == 1 ? dim_b : dim_a` covers all three valid cases: equal sizes, `a` is 1, and `b` is 1. It also gives the right answer for `[1]` with `[0]`: `dim_a` is 1, so the result is `dim_b`, which is 0.

The function takes two shapes. `TensorIterator::common_input_shape` in chapter 04 folds it over any number of inputs, starting from the first input's shape. Broadcasting is associative and commutative, so the fold order does not change the result.

## Broadcasting strides

Knowing the result shape is half of broadcasting. The other half is reading the smaller tensor as if it had the larger shape. `broadcast_strides` computes strides for that, and the key is a stride of 0.

Take a vector `v` with shape `[3]` and strides `[1]`, and read it as `[2, 3]` with strides `[0, 1]`:

```text
buffer of v:  [ 10  20  30 ]

broadcast strides: [0, 1]

           j=0   j=1   j=2
row i=0  [ 10    20    30 ]     offset = 0*i + 1*j = j
row i=1  [ 10    20    30 ]     same offsets, same data
```

Moving down a row skips 0 elements, so both rows read the same three values. The result looks like a `[2, 3]` tensor and still owns three values.

A column works the same way. Shape `[2, 1]` with strides `[1, 1]` read as `[2, 4]` gets strides `[1, 0]`: moving along a row skips 0 elements, so every column repeats the single value of its row.

The rule, per dimension and aligned from the right:

| Input size | Target size | Result stride |
|------------|-------------|---------------|
| missing (new leading dimension) | any | 0 |
| equal to target | n | original stride |
| 1 | n != 1 | 0 |
| anything else | n | error |

From `src/shape.cpp`, after the validation checks:

```cpp
const std::size_t new_dims = target.size() - shape.size();
Strides result(target.size(), 0);
for (std::size_t i = 0; i < shape.size(); ++i) {
    const std::int64_t target_dim = target[i + new_dims];
    KAEL_CHECK_VALUE(shape[i] == target_dim || shape[i] == 1, "incompatible shapes "
                                                                  << to_string(shape) << " and "
                                                                  << to_string(target));
    if (shape[i] == target_dim) {
        result[i + new_dims] = strides[i];
    }
}
return result;
```

The result starts filled with zeros, so new leading dimensions and size-1 dimensions that grow need no code: they keep the 0. Only matching dimensions copy their stride.

Before the loop, the function validates both shapes, checks that `shape` and `strides` have the same length, and checks that `shape` has no more dimensions than `target`. Broadcasting only adds dimensions. It never removes them.

A worked example with a new leading dimension. Shape `[1, 3]`, strides `[3, 1]`, target `[4, 2, 3]`:

```text
target:      4   2   3
input:           1   3
new_dims = 1

position 0: new leading dimension        -> 0
position 1: input 1, target 2            -> 0
position 2: input 3, target 3, stride 1  -> 1

strides = [0, 0, 1]
```

`broadcast_strides` does not compute the target shape. The caller computes it with `broadcast_shapes` over all inputs first, then calls `broadcast_strides` once per input. Chapter 04 shows the iterator doing that, and chapter 03 uses it for `Tensor::expand`.

A tensor with stride 0 has **internal overlap**: several logical positions map to one memory location. Reading through it is safe. Writing through it is not, because two writes land on the same element. Chapter 03 adds `Tensor::has_internal_overlap`, and the in-place operations of chapter 05 reject such tensors with a `ValueError`.

## Dimension arguments and negative dims

Many operations take a dimension argument: `sum(dims)`, `transpose(dim0, dim1)`, `size(dim)`. kael follows Python and PyTorch and accepts negative dimensions, where `-1` means the last dimension. `wrap_dim` converts them.

From `src/shape.cpp`:

```cpp
std::int64_t wrap_dim(std::int64_t dim, std::int64_t ndim) {
    KAEL_CHECK_INDEX(ndim > 0, "dimension " << dim << " specified but tensor has no dimensions");
    KAEL_CHECK_INDEX(dim >= -ndim && dim < ndim,
                     "dimension " << dim << " out of range [" << -ndim << ", " << ndim - 1 << "]");
    return dim < 0 ? dim + ndim : dim;
}

std::size_t dim_position(std::int64_t dim, std::int64_t ndim) {
    return static_cast<std::size_t>(wrap_dim(dim, ndim));
}
```

For a 3-d tensor the valid range is `[-3, 2]`, and `wrap_dim(-1, 3)` returns 2. A 0-d tensor has no dimensions at all, so every dimension argument is an error. `dim_position` returns the wrapped value as a `std::size_t`, ready to index a `Shape`, which keeps the signed-to-unsigned cast in one place instead of at every call site.

This differs from `offset_from_index` on purpose. Dimension arguments come from user code, where `-1` is idiomatic. Element indices passed to `offset_from_index` come from internal loops, where a negative value is a bug.

## Shape inference for `view` and `reshape`

`infer_shape` resolves a requested shape that may contain one `-1` against a known element count. `view({-1, 4})` on a 12-element tensor becomes `[3, 4]`.

From `src/shape.cpp`, the inference branch:

```cpp
KAEL_CHECK_VALUE(known_count != 0, "cannot infer the -1 in shape "
                                       << to_string(requested)
                                       << ": a dimension of size 0 makes it ambiguous");
KAEL_CHECK_VALUE(element_count % known_count == 0, "shape " << to_string(requested)
                                                            << " does not fit " << element_count
                                                            << " elements");

Shape result = requested;
result[static_cast<std::size_t>(std::distance(requested.begin(), inferred))] =
    element_count / known_count;
return result;
```

Before that branch, the function rejects more than one `-1`, computes `known_count` as the `numel` of the other dimensions (which also rejects any other negative value), and, when there is no `-1`, requires `known_count == element_count`.

The zero check guards an ambiguous case: `[0, -1]` for a tensor with 0 elements fits any value in place of the `-1`, so there is no single answer. PyTorch rejects the same case.

## Edge cases and pitfalls

- **Signed overflow is undefined behavior.** Every product of dimensions or strides is checked before it happens: in `numel`, in `contiguous_strides` through `checked_mul`, and in chapter 02's storage-fit check. Checking after the multiplication is wrong, because the compiler may assume the overflow does not occur.
- **Empty tensors.** `numel` returns 0 before the overflow loop. `contiguous_strides` treats size 0 as 1 so strides stay meaningful. `index_from_linear` rejects every position for an empty shape before it divides. A shape with a 0 and very large other dimensions still fails in `contiguous_strides`, because its strides cannot be represented.
- **0-d tensors.** Shape `[]` has one element, strides `[]`, and the index `[]`. Every function handles it through the general path: empty loops, product of nothing equals 1. `wrap_dim` is the exception: a 0-d tensor has no dimension to name, so it throws.
- **Unsigned countdown loops.** The loops that walk from the last dimension use `for (std::size_t i = size; i-- > 0;)`. The more common `for (std::size_t i = size - 1; i >= 0; --i)` never ends, because an unsigned value is always `>= 0`, and it wraps for an empty shape.
- **Broadcasting with size 0.** A 0 only matches a 0 or a 1. Treating 0 as "smaller, so it stretches" would be wrong and would disagree with NumPy.
- **Strides of size-1 dimensions.** `broadcast_strides` keeps the original stride when the input size equals the target size, including when both are 1. That stride is never used to step, so its value does not matter, and chapter 02's contiguity check ignores it for the same reason.
- **Typed errors.** Index problems throw `IndexError` (`offset_from_index`, `index_from_linear`, `wrap_dim`). Shape and argument problems throw `ValueError`. Code that catches `kael::Error` catches both, because both derive from it.

## How it is tested

All tests are in `tests/test_shape.cpp` and run as part of `kael_tests`. There are 46 tests in these suites:

| Suite | What it proves |
|-------|----------------|
| `ShapeString` | Formatting of normal, empty (`[]`), zero-sized and large shapes |
| `ValidateShape` | Negative dimensions throw, and the message contains the shape |
| `Numel` | Regular counts, negative dimensions, and the exact overflow boundary (`{max}` fits, `{max, 2}` throws) |
| `ContiguousStrides` | Row-major strides, 0-d shapes, size-1 dimensions, `[2, 0, 3]` gives `[3, 3, 1]` |
| `OffsetFromIndex` | Contiguous and non-contiguous strides, base offset, 0-d, out-of-range and negative indices, size mismatches, error message contents |
| `IndexFromLinear` | Row-major order, 0-d shapes, out-of-range positions, and the round trip with `offset_from_index` |
| `BroadcastShapes` | Valid and invalid pairs, symmetry, error messages, and 300 NumPy-generated cases |
| `BroadcastStrides` | Stride results, extra dimensions, mismatches, and two semantic tests that read real data |
| `WrapDim` | Positive and negative dimensions, range errors, and the 0-d error |

Three kinds of test carry most of the weight.

The round trip ties the two index functions together over every element of a 3-d shape.

From `tests/test_shape.cpp`:

```cpp
TEST(IndexFromLinear, RoundTripWithOffsetFromIndex) {
    kael::Shape shape = {2, 3, 4};
    kael::Strides strides = kael::contiguous_strides(shape);
    for (std::int64_t linear = 0; linear < kael::numel(shape); linear++) {
        kael::Shape index = kael::index_from_linear(linear, shape);
        EXPECT_EQ(kael::offset_from_index(shape, strides, 0, index), linear);
    }
}
```

With contiguous strides, the linear position and the memory offset are the same number. If either function walked the dimensions in the wrong order or had an off-by-one error, some element would map to the wrong offset.

The semantic broadcast tests (`BroadcastStrides.SemanticBroadcastRows` and `SemanticBroadcastColumns`) put real values in a buffer, broadcast a `[3]` row to `[2, 3]` and a `[2, 1]` column to `[2, 4]`, then read every element of the target through `index_from_linear` and `offset_from_index`. Each value must equal the one the broadcast is supposed to repeat. They check that stride 0 does what the diagrams in this chapter claim, not only that the stride vectors have the expected numbers.

The generated cases compare kael with NumPy. `scripts/gen_broadcast_cases.py` seeds Python's `random` with 42, draws 300 pairs of shapes with 0 to 4 dimensions and sizes from `{0, 1, 2, 3}` (weighted toward 1 and 3 so that compatible pairs are common), and asks `np.broadcast_shapes` for each answer. It writes one initializer per pair to `tests/generated/broadcast_cases.inc`:

```text
    {{1, 0}, {}, true, {1, 0}},
    {{0, 1, 0, 1}, {2, 2, 3, 3}, false, {}},
```

`BroadcastShapes.GeneratedCases` includes the file into an array and expects either the same result or a `ValueError` for every case. Regenerate the file with:

```bash
python3 scripts/gen_broadcast_cases.py
```

The output is deterministic, so a regenerated file only changes if the script changes. The generated file is committed, which means the C++ tests do not need Python or NumPy to run.

## Build it yourself

Start from an empty CMake project with a library target, GoogleTest through `FetchContent`, and the warning flags from the overview.

1. Create `include/kael/error.h` with `kael::Error` (derived from `std::runtime_error`), the subclasses `IndexError`, `ValueError` and `TypeError`, and the `KAEL_CHECK_AS` macro with the `KAEL_CHECK_INDEX` and `KAEL_CHECK_VALUE` shortcuts. Checkpoint: a test that `KAEL_CHECK_VALUE(false, "x " << 1)` throws `kael::ValueError` and the message contains `x 1`.
2. Create `include/kael/shape.h` with the `Shape` and `Strides` aliases and `to_string`. Checkpoint: the `ShapeString` tests pass.
3. Add `validate_shape` and `numel` with the zero early return and the division-based overflow check. Checkpoint: `ValidateShape.*` and `Numel.*` pass, including `numel({max})` and the throw for `numel({max, 2})`.
4. Add `src/internal/checked_math.h` with `checked_mul` and `checked_add`, then `contiguous_strides` with the `max(dim, 1)` product. Checkpoint: `ContiguousStrides.*` pass, in particular `[2, 0, 3]` gives `[3, 3, 1]`.
5. Add `offset_from_index` with `std::span` parameters, the size checks and the per-dimension range check. Checkpoint: `OffsetFromIndex.*` pass, including the non-contiguous case with strides `[1, 3]`.
6. Add `index_from_linear` with the range check before the division loop. Checkpoint: `IndexFromLinear.RoundTripWithOffsetFromIndex` passes for all 24 positions.
7. Add `broadcast_shapes`, counting positions from the right. Checkpoint: `BroadcastShapes.ValidCases`, `Symmetry` and `IncompatibleThrows` pass.
8. Add `scripts/gen_broadcast_cases.py`, run it, and add `BroadcastShapes.GeneratedCases`. Checkpoint: all 300 cases agree with NumPy.
9. Add `broadcast_strides` with the zero-filled result. Checkpoint: `BroadcastStrides.SemanticBroadcastRows` and `SemanticBroadcastColumns` pass.
10. Add `wrap_dim`, `dim_position` and `infer_shape`. They are only needed from chapter 02 and 03 on, but they belong in this file. Checkpoint: `WrapDim.*` pass, and `infer_shape({-1, 4}, 12)` returns `[3, 4]` while `infer_shape({0, -1}, 0)` throws.
11. Build with `-DKAEL_ENABLE_SANITIZERS=ON` and run the suite again. Checkpoint: no AddressSanitizer or UndefinedBehaviorSanitizer reports. UBSan reports signed overflow, so this run confirms that no overflow check was skipped.

## Further reading

- The NumPy documentation on broadcasting and on the strided memory layout of `ndarray`.
- Edward Z. Yang, "PyTorch internals", the section on strides and views.
- The C++ standard rules on signed integer overflow, and the UndefinedBehaviorSanitizer documentation for `-fsanitize=signed-integer-overflow`.

Next: [02. Memory model](02-memory-model.md)
