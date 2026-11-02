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
