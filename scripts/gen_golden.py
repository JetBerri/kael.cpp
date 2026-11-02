import shutil
import zlib
from pathlib import Path

import numpy as np
import torch

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden"
DTYPES = {"f32": torch.float32, "f64": torch.float64, "i64": torch.int64, "bool": torch.bool}
SPECIAL_VALUES = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0, 1e-10, 1000.0, -1000.0]


def spec(shape, dtype="f32", kind="normal", view="none", values=None):
    return {"shape": shape, "dtype": dtype, "kind": kind, "view": view, "values": values}


def make_base(input_spec):
    dtype = DTYPES[input_spec["dtype"]]
    shape = input_spec["shape"]
    if input_spec["values"] is not None:
        return torch.tensor(input_spec["values"], dtype=dtype).reshape(shape)
    if dtype == torch.bool:
        return torch.rand(shape) > 0.5
    if dtype == torch.int64:
        return torch.randint(-5, 6, shape, dtype=dtype)
    if input_spec["kind"] == "positive":
        return torch.rand(shape, dtype=dtype) * 3 + 0.01
    if input_spec["kind"] == "index":
        raise ValueError("index inputs must give explicit values")
    return torch.randn(shape, dtype=dtype)


def apply_view(t, view):
    name, *raw = view.split(":")
    args = [int(a) for a in raw]
    if name == "none":
        return t
    if name == "transpose":
        return t.transpose(args[0], args[1])
    if name == "permute":
        return t.permute(args)
    if name == "slice":
        dim, start, end, step = args
        key = [slice(None)] * t.dim()
        key[dim] = slice(start, end, step)
        return t[tuple(key)]
    if name == "expand":
        return t.expand(args)
    raise ValueError(view)


def encode_args(args):
    def encode(value):
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, (list, tuple)):
            return "/".join(str(v) for v in value)
        return str(value)

    return ",".join(f"{key}:{encode(value)}" for key, value in args.items())


def write_case(manifest, op, name, inputs, args, compute):
    torch.manual_seed(zlib.crc32(f"{op}/{name}".encode()))
    directory = GOLDEN_DIR / op
    directory.mkdir(parents=True, exist_ok=True)

    viewed = []
    for input_name, input_spec in inputs.items():
        base = make_base(input_spec)
        np.save(directory / f"{name}_{input_name}.npy", base.numpy())
        viewed.append(apply_view(base, input_spec["view"]))

    expected = compute(*viewed, **args)
    np.save(directory / f"{name}_out.npy", expected.contiguous().numpy())

    views = " ".join(f"view_{key}={value['view']}" for key, value in inputs.items())
    manifest.append(
        f"op={op} case={name} inputs={','.join(inputs)} {views} args={encode_args(args)}"
    )


def write_npy_fixtures():
    directory = GOLDEN_DIR / "npy"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "f32.npy", np.arange(6, dtype=np.float32).reshape(2, 3) * 1.5)
    np.save(directory / "f64.npy", np.array([0.25, 0.5, 0.75], dtype=np.float64))
    np.save(directory / "i64.npy", np.array([-1, 2], dtype=np.int64))
    np.save(directory / "bool.npy", np.array([True, False, True]))
    np.save(directory / "scalar.npy", np.array(2.5, dtype=np.float32))
    np.save(directory / "empty.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(directory / "fortran.npy", np.asfortranarray(np.arange(6.0).reshape(2, 3)))
    np.save(directory / "f16.npy", np.zeros(2, dtype=np.float16))
    (directory / "not_npy.npy").write_bytes(b"this is not an npy file")


CASES = []


def add_cases(op, compute, patterns):
    for name, inputs, args in patterns:
        CASES.append((op, name, inputs, args, compute))


SPECIAL_A = [v for v in SPECIAL_VALUES for _ in SPECIAL_VALUES]
SPECIAL_B = [v for _ in SPECIAL_VALUES for v in SPECIAL_VALUES]

BROADCAST_PATTERNS = [
    ("same", {"a": spec((3, 4)), "b": spec((3, 4))}, {}),
    ("broadcast_rank", {"a": spec((2, 3, 4)), "b": spec((4,))}, {}),
    ("broadcast_ones", {"a": spec((3, 1)), "b": spec((1, 4))}, {}),
    ("transposed", {"a": spec((4, 3), view="transpose:0:1"), "b": spec((3, 4))}, {}),
    ("stepped", {"a": spec((3, 8), view="slice:1:0:8:2"), "b": spec((3, 4))}, {}),
    ("expanded", {"a": spec((4,), view="expand:3:4"), "b": spec((3, 4))}, {}),
    ("permuted_broadcast", {"a": spec((4, 2, 3), view="permute:1:2:0"), "b": spec((3, 1))}, {}),
    ("scalar_tensor", {"a": spec(()), "b": spec((3,))}, {}),
    ("scalar_scalar", {"a": spec(()), "b": spec(())}, {}),
    ("empty", {"a": spec((0, 3)), "b": spec((3,))}, {}),
    ("f64", {"a": spec((3, 4), "f64"), "b": spec((3, 4), "f64")}, {}),
]
PROMOTION_PATTERNS = [
    ("f32_i64", {"a": spec((5,)), "b": spec((5,), "i64")}, {}),
    ("f64_f32", {"a": spec((5,), "f64"), "b": spec((5,))}, {}),
    ("i64_i64", {"a": spec((5,), "i64"), "b": spec((5,), "i64")}, {}),
]
SPECIAL_PATTERN = [
    ("special", {"a": spec((100,), values=SPECIAL_A), "b": spec((100,), values=SPECIAL_B)}, {}),
]
BOOL_PATTERNS = [
    ("bools", {"a": spec((6,), "bool"), "b": spec((6,), "bool")}, {}),
    ("bool_f32", {"a": spec((6,), "bool"), "b": spec((6,))}, {}),
]
ALL_BINARY = BROADCAST_PATTERNS + PROMOTION_PATTERNS + SPECIAL_PATTERN

add_cases("add", torch.add, ALL_BINARY + BOOL_PATTERNS[1:])
add_cases("sub", torch.sub, ALL_BINARY)
add_cases("mul", torch.mul, ALL_BINARY + BOOL_PATTERNS)
add_cases("div", torch.div, ALL_BINARY + BOOL_PATTERNS)
add_cases("maximum", torch.maximum, ALL_BINARY)
add_cases("minimum", torch.minimum, ALL_BINARY)
add_cases(
    "floor_divide",
    torch.floor_divide,
    BROADCAST_PATTERNS
    + [
        (
            "i64_nonzero",
            {
                "a": spec((6,), "i64", values=[7, -7, 7, -7, 0, 5]),
                "b": spec((6,), "i64", values=[2, 2, -2, -2, 3, 7]),
            },
            {},
        )
    ],
)
add_cases(
    "pow",
    torch.pow,
    [
        ("positive_base", {"a": spec((3, 4), kind="positive"), "b": spec((3, 4))}, {}),
        ("broadcast", {"a": spec((2, 3), kind="positive"), "b": spec((3,))}, {}),
        ("transposed", {"a": spec((4, 3), kind="positive", view="transpose:0:1"), "b": spec((3, 4))}, {}),
        (
            "i64",
            {
                "a": spec((4,), "i64", values=[2, -3, 5, 0]),
                "b": spec((4,), "i64", values=[10, 3, 0, 4]),
            },
            {},
        ),
    ],
)
for comparison in [torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge]:
    add_cases(comparison.__name__, comparison, ALL_BINARY + BOOL_PATTERNS)
for logical in [torch.logical_and, torch.logical_or, torch.logical_xor]:
    add_cases(logical.__name__, logical, ALL_BINARY + BOOL_PATTERNS)

SPECIAL_UNARY = ("special", {"x": spec((10,), values=SPECIAL_VALUES)}, {})
UNARY_PATTERNS = [
    ("matrix", {"x": spec((3, 4))}, {}),
    ("transposed", {"x": spec((4, 3), view="transpose:0:1")}, {}),
    ("stepped", {"x": spec((6,), view="slice:0:0:6:2")}, {}),
    ("expanded", {"x": spec((3,), view="expand:2:3")}, {}),
    ("scalar", {"x": spec(())}, {}),
    ("empty", {"x": spec((0, 2))}, {}),
    ("f64", {"x": spec((3, 2), "f64")}, {}),
    SPECIAL_UNARY,
]
POSITIVE_PATTERNS = [
    ("matrix", {"x": spec((3, 4), kind="positive")}, {}),
    ("transposed", {"x": spec((4, 3), kind="positive", view="transpose:0:1")}, {}),
    ("f64", {"x": spec((3, 2), "f64", kind="positive")}, {}),
    SPECIAL_UNARY,
]
INTEGER_INPUT = ("i64", {"x": spec((5,), "i64", values=[1, 2, 3, 4, 5])}, {})

for function in [torch.neg, torch.abs, torch.sign]:
    add_cases(function.__name__, function, UNARY_PATTERNS + [INTEGER_INPUT])
for function in [torch.exp, torch.expm1, torch.sin, torch.cos, torch.tanh, torch.sigmoid]:
    add_cases(function.__name__, function, UNARY_PATTERNS + [INTEGER_INPUT])
for function in [torch.log, torch.log1p, torch.sqrt, torch.rsqrt]:
    add_cases(function.__name__, function, POSITIVE_PATTERNS + [INTEGER_INPUT])
add_cases("relu", torch.relu, UNARY_PATTERNS)
add_cases("gelu", torch.nn.functional.gelu, UNARY_PATTERNS)
add_cases(
    "clamp",
    torch.clamp,
    [
        (name, inputs, {"min": -0.5, "max": 0.5})
        for name, inputs, _ in UNARY_PATTERNS + [INTEGER_INPUT]
    ],
)

INPLACE_PATTERNS = [
    ("same", {"a": spec((3, 4)), "b": spec((3, 4))}, {}),
    ("broadcast_other", {"a": spec((2, 3)), "b": spec((3,))}, {}),
    ("transposed_self", {"a": spec((4, 3), view="transpose:0:1"), "b": spec((3, 4))}, {}),
    ("stepped_self", {"a": spec((3, 8), view="slice:1:0:8:2"), "b": spec((4,))}, {}),
    ("f32_i64", {"a": spec((5,)), "b": spec((5,), "i64")}, {}),
    ("f64", {"a": spec((3,), "f64"), "b": spec((3,), "f64")}, {}),
]
add_cases("add_", lambda a, b: a.add_(b), INPLACE_PATTERNS)
add_cases("sub_", lambda a, b: a.sub_(b), INPLACE_PATTERNS)
add_cases("mul_", lambda a, b: a.mul_(b), INPLACE_PATTERNS)
add_cases("div_", lambda a, b: a.div_(b), INPLACE_PATTERNS)
add_cases("copy_", lambda a, b: a.copy_(b), INPLACE_PATTERNS)
add_cases(
    "pow_",
    lambda a, b: a.pow_(b),
    [
        ("same", {"a": spec((3, 4), kind="positive"), "b": spec((3, 4))}, {}),
        ("transposed_self", {"a": spec((4, 3), kind="positive", view="transpose:0:1"), "b": spec((3, 4))}, {}),
    ],
)
add_cases(
    "clamp_",
    lambda a, min, max: a.clamp_(min, max),
    [
        ("transposed", {"a": spec((4, 3), view="transpose:0:1")}, {"min": -0.5, "max": 0.5}),
        ("stepped", {"a": spec((6,), view="slice:0:0:6:2")}, {"min": -0.5, "max": 0.5}),
    ],
)

REDUCTION_PATTERNS = [
    ("all_dims", {"x": spec((2, 3, 4))}, {}),
    ("dim0", {"x": spec((2, 3, 4))}, {"dims": [0]}),
    ("dim1", {"x": spec((2, 3, 4))}, {"dims": [1]}),
    ("dim2", {"x": spec((2, 3, 4))}, {"dims": [2]}),
    ("negative_dim", {"x": spec((2, 3, 4))}, {"dims": [-2]}),
    ("two_dims", {"x": spec((2, 3, 4))}, {"dims": [0, 2]}),
    ("keepdim", {"x": spec((2, 3, 4))}, {"dims": [1], "keepdim": True}),
    ("keepdim_two", {"x": spec((2, 3, 4))}, {"dims": [1, 2], "keepdim": True}),
    ("keepdim_all", {"x": spec((2, 3))}, {"keepdim": True}),
    ("transposed", {"x": spec((4, 3), view="transpose:0:1")}, {"dims": [1]}),
    ("stepped", {"x": spec((3, 8), view="slice:1:0:8:2")}, {"dims": [0]}),
    ("expanded", {"x": spec((3,), view="expand:2:3")}, {"dims": [0]}),
    ("permuted", {"x": spec((2, 3, 4), view="permute:2:0:1")}, {"dims": [1]}),
    ("scalar", {"x": spec(())}, {}),
    ("empty_kept", {"x": spec((0, 3))}, {"dims": [1]}),
    ("f64", {"x": spec((3, 4), "f64")}, {"dims": [1]}),
    ("nan", {"x": spec((5,), values=[1.0, float("nan"), 3.0, -1.0, 0.5])}, {}),
]
INTEGER_REDUCTIONS = [
    ("i64", {"x": spec((3, 4), "i64")}, {"dims": [0]}),
    ("i64_all", {"x": spec((3, 4), "i64")}, {}),
]
BOOL_REDUCTIONS = [
    ("bool", {"x": spec((3, 4), "bool")}, {"dims": [1]}),
    ("bool_all", {"x": spec((3, 4), "bool")}, {}),
]
EMPTY_ALL = ("empty_all", {"x": spec((0, 3))}, {})


def over_dims(function):
    def compute(x, dims=None, keepdim=False, **extra):
        return function(x, dim=list(range(x.dim())) if dims is None else dims, keepdim=keepdim, **extra)

    return compute


add_cases("sum", over_dims(torch.sum), REDUCTION_PATTERNS + INTEGER_REDUCTIONS + BOOL_REDUCTIONS + [EMPTY_ALL])
add_cases(
    "prod",
    lambda x, dims=None, keepdim=False: (
        x.prod() if dims is None else x.prod(dim=dims[0], keepdim=keepdim)
    ),
    [p for p in REDUCTION_PATTERNS if len(p[2].get("dims", [])) == 1 or "keepdim" not in p[2] and "dims" not in p[2]],
)
add_cases("mean", over_dims(torch.mean), REDUCTION_PATTERNS + [EMPTY_ALL])
add_cases("max", over_dims(torch.amax), REDUCTION_PATTERNS + INTEGER_REDUCTIONS)
add_cases("min", over_dims(torch.amin), REDUCTION_PATTERNS + INTEGER_REDUCTIONS)
add_cases("all", over_dims(torch.all), REDUCTION_PATTERNS + BOOL_REDUCTIONS + [EMPTY_ALL])
add_cases("any", over_dims(torch.any), REDUCTION_PATTERNS + BOOL_REDUCTIONS + [EMPTY_ALL])
add_cases("var", over_dims(torch.var), REDUCTION_PATTERNS)
add_cases("std", over_dims(torch.std), REDUCTION_PATTERNS)
add_cases(
    "var_correction0",
    lambda x, dims=None, keepdim=False: over_dims(torch.var)(x, dims, keepdim, correction=0),
    REDUCTION_PATTERNS,
)
add_cases(
    "norm",
    lambda x, dims=None, keepdim=False: torch.linalg.vector_norm(x, dim=dims, keepdim=keepdim),
    [p for p in REDUCTION_PATTERNS if p[0] not in ("scalar", "keepdim_all", "empty_kept")],
)
ARG_PATTERNS = [
    ("flat", {"x": spec((2, 3, 4))}, {}),
    ("dim0", {"x": spec((2, 3, 4))}, {"dim": 0}),
    ("dim_negative", {"x": spec((2, 3, 4))}, {"dim": -1}),
    ("keepdim", {"x": spec((2, 3, 4))}, {"dim": 1, "keepdim": True}),
    ("transposed", {"x": spec((4, 3), view="transpose:0:1")}, {"dim": 1}),
    ("ties", {"x": spec((6,), "i64", values=[1, 3, 3, 0, 0, 3])}, {}),
    ("nan", {"x": spec((4,), values=[1.0, float("nan"), 3.0, float("nan")])}, {}),
    ("scalar", {"x": spec(())}, {}),
]
add_cases("argmax", lambda x, dim=None, keepdim=False: torch.argmax(x, dim=dim, keepdim=keepdim), ARG_PATTERNS)
add_cases("argmin", lambda x, dim=None, keepdim=False: torch.argmin(x, dim=dim, keepdim=keepdim), ARG_PATTERNS)

MATMUL_PATTERNS = [
    ("vector_vector", {"a": spec((4,)), "b": spec((4,))}, {}),
    ("matrix_matrix", {"a": spec((2, 3)), "b": spec((3, 4))}, {}),
    ("vector_matrix", {"a": spec((3,)), "b": spec((3, 2))}, {}),
    ("matrix_vector", {"a": spec((2, 3)), "b": spec((3,))}, {}),
    ("batch_same", {"a": spec((5, 2, 3)), "b": spec((5, 3, 2))}, {}),
    ("batch_broadcast", {"a": spec((2, 1, 3, 4)), "b": spec((5, 4, 2))}, {}),
    ("batch_vector", {"a": spec((3,)), "b": spec((5, 3, 2))}, {}),
    ("batch_matrix_vector", {"a": spec((5, 2, 3)), "b": spec((3,))}, {}),
    ("transposed", {"a": spec((3, 2), view="transpose:0:1"), "b": spec((4, 3), view="transpose:0:1")}, {}),
    ("stepped", {"a": spec((2, 6), view="slice:1:0:6:2"), "b": spec((3, 2))}, {}),
    ("expanded", {"a": spec((3,), view="expand:2:3"), "b": spec((3, 2))}, {}),
    ("empty_inner", {"a": spec((2, 0)), "b": spec((0, 3))}, {}),
    ("empty_batch", {"a": spec((0, 2, 3)), "b": spec((3, 2))}, {}),
    ("f64", {"a": spec((2, 3), "f64"), "b": spec((3, 2), "f64")}, {}),
    ("i64", {"a": spec((2, 3), "i64"), "b": spec((3, 2), "i64")}, {}),
]
add_cases("matmul", torch.matmul, MATMUL_PATTERNS)
add_cases("dot", torch.dot, MATMUL_PATTERNS[:1])
add_cases(
    "outer",
    torch.outer,
    [
        ("plain", {"a": spec((3,)), "b": spec((4,))}, {}),
        ("stepped", {"a": spec((6,), view="slice:0:0:6:2"), "b": spec((2,))}, {}),
    ],
)

add_cases(
    "cat",
    lambda a, b, dim: torch.cat([a, b], dim=dim),
    [
        ("dim0", {"a": spec((2, 3)), "b": spec((1, 3))}, {"dim": 0}),
        ("dim1", {"a": spec((2, 3)), "b": spec((2, 2))}, {"dim": 1}),
        ("negative", {"a": spec((2, 3)), "b": spec((2, 2))}, {"dim": -1}),
        ("transposed", {"a": spec((3, 2), view="transpose:0:1"), "b": spec((2, 2))}, {"dim": 1}),
        ("stepped", {"a": spec((6,), view="slice:0:0:6:2"), "b": spec((2,))}, {"dim": 0}),
        ("promotes", {"a": spec((2,)), "b": spec((2,), "i64")}, {"dim": 0}),
        ("empty_part", {"a": spec((0, 3)), "b": spec((2, 3))}, {"dim": 0}),
    ],
)
add_cases(
    "stack",
    lambda a, b, dim: torch.stack([a, b], dim=dim),
    [
        ("dim0", {"a": spec((2, 3)), "b": spec((2, 3))}, {"dim": 0}),
        ("dim_last", {"a": spec((2, 3)), "b": spec((2, 3))}, {"dim": 2}),
        ("negative", {"a": spec((2, 3)), "b": spec((2, 3))}, {"dim": -1}),
        ("transposed", {"a": spec((3, 2), view="transpose:0:1"), "b": spec((2, 3))}, {"dim": 1}),
    ],
)
add_cases(
    "where",
    torch.where,
    [
        ("same", {"c": spec((3, 4), "bool"), "a": spec((3, 4)), "b": spec((3, 4))}, {}),
        ("broadcast_all", {"c": spec((2, 1), "bool"), "a": spec((3,)), "b": spec((2, 3))}, {}),
        ("transposed", {"c": spec((4, 3), "bool", view="transpose:0:1"), "a": spec((3, 4)), "b": spec(())}, {}),
        ("promotes", {"c": spec((3,), "bool"), "a": spec((3,), "i64"), "b": spec((3,))}, {}),
        ("empty", {"c": spec((0,), "bool"), "a": spec((0,)), "b": spec((0,))}, {}),
    ],
)
add_cases(
    "masked_fill",
    lambda x, m, value: x.masked_fill(m, value),
    [
        ("same", {"x": spec((3, 4)), "m": spec((3, 4), "bool")}, {"value": 7.5}),
        ("broadcast_mask", {"x": spec((2, 3)), "m": spec((3,), "bool")}, {"value": -1.0}),
        ("transposed", {"x": spec((4, 3), view="transpose:0:1"), "m": spec((3, 4), "bool")}, {"value": 2.0}),
        ("minus_infinity", {"x": spec((3,)), "m": spec((3,), "bool")}, {"value": -float("inf")}),
        ("i64", {"x": spec((4,), "i64"), "m": spec((4,), "bool")}, {"value": 9}),
    ],
)
INDICES_1D = spec((4,), "i64", values=[2, 0, 2, 1])
add_cases(
    "index_select",
    lambda x, i, dim: torch.index_select(x, dim, i),
    [
        ("dim0", {"x": spec((3, 4)), "i": INDICES_1D}, {"dim": 0}),
        ("dim1", {"x": spec((2, 3)), "i": INDICES_1D}, {"dim": 1}),
        ("negative", {"x": spec((2, 3)), "i": INDICES_1D}, {"dim": -1}),
        ("transposed", {"x": spec((4, 3), view="transpose:0:1"), "i": INDICES_1D}, {"dim": 0}),
        ("empty", {"x": spec((3, 4)), "i": spec((0,), "i64", values=[])}, {"dim": 0}),
    ],
)
GATHER_INDEX = spec((2, 3), "i64", values=[0, 1, 0, 1, 1, 0])
add_cases(
    "gather",
    lambda x, i, dim: torch.gather(x, dim, i),
    [
        ("dim0", {"x": spec((2, 3)), "i": GATHER_INDEX}, {"dim": 0}),
        ("dim1", {"x": spec((2, 4)), "i": GATHER_INDEX}, {"dim": 1}),
        ("negative", {"x": spec((2, 4)), "i": GATHER_INDEX}, {"dim": -1}),
        ("transposed", {"x": spec((3, 2), view="transpose:0:1"), "i": GATHER_INDEX}, {"dim": 1}),
        ("smaller_index", {"x": spec((5, 4)), "i": GATHER_INDEX}, {"dim": 1}),
        ("i64", {"x": spec((2, 3), "i64"), "i": GATHER_INDEX}, {"dim": 1}),
    ],
)
add_cases(
    "scatter_add_",
    lambda x, i, s, dim: x.scatter_add_(dim, i, s),
    [
        ("dim1_repeated", {"x": spec((2, 4)), "i": GATHER_INDEX, "s": spec((2, 3))}, {"dim": 1}),
        ("dim0", {"x": spec((2, 3)), "i": GATHER_INDEX, "s": spec((2, 3))}, {"dim": 0}),
        ("larger_source", {"x": spec((2, 4)), "i": GATHER_INDEX, "s": spec((3, 5))}, {"dim": 1}),
    ],
)

NN_PATTERNS = [
    ("dim0", {"x": spec((3, 4))}, {"dim": 0}),
    ("dim1", {"x": spec((3, 4))}, {"dim": 1}),
    ("negative", {"x": spec((2, 3, 4))}, {"dim": -1}),
    ("transposed", {"x": spec((4, 3), view="transpose:0:1")}, {"dim": 1}),
    ("stepped", {"x": spec((3, 8), view="slice:1:0:8:2")}, {"dim": 1}),
    ("large_values", {"x": spec((4,), values=[1000.0, 1000.0, -1000.0, 999.0])}, {"dim": 0}),
    ("some_minus_infinity", {"x": spec((3,), values=[-float("inf"), 0.0, 1.0])}, {"dim": 0}),
    ("empty", {"x": spec((0, 3))}, {"dim": 1}),
    ("scalar", {"x": spec(())}, {"dim": 0}),
    ("f64", {"x": spec((3, 4), "f64")}, {"dim": 1}),
]
ALL_MINUS_INFINITY = ("all_minus_infinity", {"x": spec((2,), values=[-float("inf"), -float("inf")])}, {"dim": 0})
PLUS_INFINITY = ("plus_infinity", {"x": spec((2,), values=[float("inf"), 1.0])}, {"dim": 0})
add_cases(
    "logsumexp",
    torch.logsumexp,
    NN_PATTERNS
    + [ALL_MINUS_INFINITY, PLUS_INFINITY, ("keepdim", {"x": spec((3, 4))}, {"dim": 1, "keepdim": True})],
)
add_cases("softmax", torch.softmax, NN_PATTERNS)
add_cases("log_softmax", torch.log_softmax, NN_PATTERNS)


def main():
    if GOLDEN_DIR.exists():
        shutil.rmtree(GOLDEN_DIR)
    write_npy_fixtures()
    manifest = []
    for op, name, inputs, args, compute in CASES:
        write_case(manifest, op, name, inputs, args, compute)
    (GOLDEN_DIR / "manifest.txt").write_text("".join(line + "\n" for line in manifest))
    print(f"Generated {len(manifest)} cases in {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
