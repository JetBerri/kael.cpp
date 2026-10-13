# 01. Shapes and strides

This chapter covers phase 2 of kael: the metadata that turns a flat block of memory into a tensor. There is no `Tensor` class yet. Everything here is a handful of free functions in `include/kael/shape.h`, and they are the foundation that every later phase builds on.

If you can program but have never looked inside a tensor, this is for you.

## A tensor is a buffer plus metadata

In memory, a tensor is nothing special. It is a flat array of numbers plus three pieces of metadata that say how to read it:

- **shape**: how many elements along each dimension, for example `[2, 3]` for a matrix with 2 rows and 3 columns.
- **strides**: how many positions to jump in the flat array to move one step along each dimension.
- **offset**: where the first element lives inside the buffer.

The buffer stores the data. The metadata stores the interpretation. Keeping them separate is what lets us later reshape, transpose and broadcast without touching a single byte of data.

```
buffer:  [ 1.0  2.0  3.0  4.0  5.0  6.0 ]

shape:   [2, 3]
strides: [3, 1]
offset:  0
```

In kael both `Shape` and `Strides` are just `std::vector<std::int64_t>`.

## Shape and numel

The shape is the list of dimension sizes. `numel` is the number of elements, which is the product of all of them.

```
shape        numel
[2, 3, 4]    24
[5]          5
[]           1
[3, 0, 2]    0
```

Two cases deserve a comment.

**The scalar.** A shape with no dimensions, `[]`, is a scalar. The product of an empty list is 1, so it holds exactly one element. This is not a special case we chose, it falls out of the math.

**A zero dimension.** A shape like `[3, 0, 2]` has 0 elements. It is a legal tensor, just an empty one. Empty tensors show up naturally, for example a batch with no samples, so kael allows them instead of throwing.

### The overflow check

`numel` multiplies dimensions together, and with `int64_t` that product can overflow. Signed overflow in C++ is undefined behaviour, so we cannot multiply first and check afterwards. The result would be garbage and the compiler is free to assume it never happened.

Instead, before each multiplication, we check that the running total still fits:

```cpp
KAEL_CHECK(total <= max / dim, "number of elements overflows int64 ...");
total *= dim;
```

If `total * dim` would exceed the maximum, then `total > max / dim`, and we throw before the multiplication happens. Any dimension equal to 0 is handled first by returning 0, so the division is always safe.

## Row-major layout and strides

A 2 by 3 matrix is stored in memory one row after another. This is called row-major order and it is what C, NumPy and PyTorch use by default.

```
logical view (shape [2, 3]):

        col 0   col 1   col 2
row 0 [  a       b       c   ]
row 1 [  d       e       f   ]

flat memory:

index:   0    1    2    3    4    5
value: [ a    b    c    d    e    f ]
```

To move one column to the right you jump 1 position in memory. To move one row down you jump 3 positions, because you have to skip a whole row. Those jumps are the strides: `[3, 1]`.

### Computing contiguous strides

The strides for a plain, row-major layout are built from right to left with an accumulated product:

```
shape [2, 3, 4]

start with acc = 1
dim 2 (size 4): stride = 1    then acc = 1 * 4 = 4
dim 1 (size 3): stride = 4    then acc = 4 * 3 = 12
dim 0 (size 2): stride = 12   then acc = 12 * 2 = 24

strides = [12, 4, 1]
```

The last dimension always has stride 1 because its elements sit next to each other. Each dimension to the left has to skip everything to its right.

### Why `max(dim, 1)`

When accumulating, kael multiplies by `max(dim, 1)` instead of `dim`:

```cpp
acc *= std::max<std::int64_t>(shape[idx], 1);
```

A zero-size dimension has no elements, so its stride is never used to read data. But if we multiplied by 0, every stride to its left would become 0 too, and the tensor would lose the information about how its other dimensions are laid out. Treating 0 as 1 keeps the strides well formed, which matters later when we take views of empty tensors. NumPy does the same.

```
shape [2, 0, 3]  ->  strides [3, 3, 1]   (not [0, 0, 1])
```

## From index to offset, and back

### offset_from_index

Given a multidimensional index, the position in the flat buffer is:

```
offset = base_offset + sum(index[i] * strides[i])
```

Step by step for shape `[2, 3, 4]`, strides `[12, 4, 1]`, base offset 0, index `[1, 0, 1]`:

```
dimension 0:  1 * 12 = 12
dimension 1:  0 *  4 =  0
dimension 2:  1 *  1 =  1
                     -----
offset             =  13
```

Element `[1, 0, 1]` is at position 13 in the buffer. The function also checks that each index is inside `[0, shape[i])` and throws a `kael::Error` otherwise.

### index_from_linear

The inverse operation. Given a position in row-major order, recover the multidimensional index. We walk the shape from right to left: the remainder is the coordinate for this dimension, the quotient is what is left for the dimensions to its left.

Same example, linear 13 in shape `[2, 3, 4]`:

```
dimension 2 (size 4):  13 % 4 = 1    remaining 13 / 4 = 3
dimension 1 (size 3):   3 % 3 = 0    remaining  3 / 3 = 1
dimension 0 (size 2):   1 % 2 = 1    remaining  1 / 2 = 0

index = [1, 0, 1]
```

We got back the index we started with.

### The round-trip test

The test `IndexFromLinear.RoundTripWithOffsetFromIndex` does exactly this for every element of a `[2, 3, 4]` tensor:

```cpp
for (linear = 0; linear < numel(shape); linear++) {
    index = index_from_linear(linear, shape);
    EXPECT_EQ(offset_from_index(shape, strides, 0, index), linear);
}
```

What it guarantees: with contiguous strides, the two functions are exact inverses over the whole range. Every linear position maps to one index and that index maps back to the same position. If either function had an off-by-one error or walked the dimensions in the wrong order, this test would catch it.

## Transposing without copying

Here is the payoff of keeping data and metadata separate. Take the same 6 numbers:

```
buffer: [ a  b  c  d  e  f ]
```

Read with shape `[2, 3]` and strides `[3, 1]`:

```
[ a  b  c ]
[ d  e  f ]
```

Read with shape `[3, 2]` and strides `[1, 3]`:

```
[ a  d ]
[ b  e ]
[ c  f ]
```

That second one is the transpose, and we did not move anything. Check with the formula: element `[2, 1]` of the transposed view is at `2 * 1 + 1 * 3 = 5`, which is `f`. In the original, `f` is at `[1, 2]`, so the transpose is correct.

Swapping the shape and the strides is all a transpose is. This is what phase 4 (views) will build on.

## Broadcasting

Broadcasting is how a `[3]` vector can be added to a `[2, 3]` matrix without first making a copy with two rows. It is a rule for combining two shapes into one.

### The three rules

1. Align the shapes from the right. If one has fewer dimensions, treat the missing leading ones as 1.
2. Two dimensions are compatible if they are equal, or if one of them is 1.
3. The result takes the larger of the two in each position. If they are equal, it takes that value.

If any pair of dimensions fails rule 2, the shapes are not broadcastable and `broadcast_shapes` throws.

### An example, aligned from the right

```
a:       [5, 1, 4]
b:          [3, 1]      <- missing leading dim counts as 1

aligned:
a:        5   1   4
b:        1   3   1
          |   |   |
result:   5   3   4
```

Position by position: 5 vs 1 gives 5, 1 vs 3 gives 3, 4 vs 1 gives 4. Result `[5, 3, 4]`.

A dimension of size 0 follows the same rules. `[0]` with `[1]` gives `[0]`, but `[0]` with `[3]` is an error, exactly like NumPy.

### Stride 0: broadcasting without copying

Knowing the result shape is half the story. The other half is reading the smaller tensor as if it had the bigger shape. That is what `broadcast_strides` does, and the trick is a stride of 0.

Take `v` with shape `[3]`, strides `[1]`, and broadcast it to `[2, 3]`:

```
buffer of v:  [ 10  20  30 ]

broadcast strides: [0, 1]

           j=0   j=1   j=2
row i=0  [ 10    20    30 ]     offset = 0*i + 1*j = j
row i=1  [ 10    20    30 ]     same offsets, same data
```

Moving down a row jumps 0 positions, so every row reads the same three numbers. The tensor looks like `[2, 3]` but still owns only 3 values.

The rule inside `broadcast_strides` is simple. For each dimension, aligned from the right:

- if the sizes match, keep the original stride
- if the original size is 1, use stride 0
- any missing leading dimension also gets stride 0
- anything else is an error

## How this was tested

The suite has 43 tests in `tests/test_shape.cpp`, in four groups.

**Unit tests per function.** Regular cases, the scalar, zero dimensions, negative dimensions, out-of-range indices, mismatched dimension counts, and the int64 overflow boundary. Several tests also check that the error message contains the offending shape and position, so a failure is readable.

**The round-trip test.** Described above. It covers every element of a 3D tensor and ties `offset_from_index` and `index_from_linear` together.

**Semantic broadcast tests.** Instead of only comparing stride vectors, these put real data in a buffer, broadcast a `[3]` row and a `[2, 1]` column, and then read every element of the target through `index_from_linear` and `offset_from_index`. Each value read must equal the value the broadcast is supposed to repeat. This checks that stride 0 actually does what the diagram above claims.

**NumPy-generated cases.** The script `scripts/gen_broadcast_cases.py` draws 300 random shape pairs with a fixed seed, asks `np.broadcast_shapes` for the answer, and writes them to `tests/generated/broadcast_cases.inc`. The C++ test runs all of them and expects the same result or the same failure. This is how we know the broadcasting rules match NumPy, including the odd cases with zeros and ones.

## Decisions

**Zero dimensions are allowed.** Empty tensors are legal, `numel` returns 0, and strides stay well formed. Only negative dimensions are an error.

**`int64_t` instead of `int`.** Shapes, strides and offsets can get large, and products of dimensions get large faster. A 32-bit `int` would overflow on tensors that are perfectly normal today. Using 64 bits everywhere also avoids mixing signed widths in arithmetic.

**No negative indices at this layer.** Python lets you write `-1` to mean the last element. That is convenient at the top, but at this level it would hide bugs: an index that went negative by mistake would silently wrap instead of failing. These functions accept only `[0, shape[i])`. If we want Python-style indexing, it will be sugar on top, not a rule down here.

**`kael::Error` and `KAEL_CHECK`.** Every error in kael is a `kael::Error`, a thin subclass of `std::runtime_error`. The `KAEL_CHECK(cond, msg)` macro evaluates a condition and, on failure, throws with the file, the line, the text of the condition and a streamed message:

```
kael error at src/shape.cpp:25: check 'shape[i] >= 0' failed:
negative dimension: -2 at position 1 in shape: [2, -2]
```

The message argument is deliberately not parenthesised, so you can write `"index " << i << " out of range"` directly. Having one error type and one macro means every later phase gets the same readable failures for free.

## Next chapter

Phase 3 puts this to use: a `Storage` that owns the buffer and a `Tensor` that pairs it with a shape, strides and an offset.
