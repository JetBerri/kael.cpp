import builtins as _builtins
import functools as _functools

from . import _C
from ._C import (
    DType,
    GPT,
    Tensor,
    __version__,
    abs,
    add,
    all,
    allocator_stats,
    any,
    argmax,
    argmin,
    autograd_graph,
    bmm,
    cat,
    clamp,
    cos,
    cuda_is_available,
    cuda_synchronize,
    cumsum,
    div,
    dot,
    empty_cache,
    empty_like,
    eq,
    equal,
    exp,
    explain,
    expm1,
    floor_divide,
    from_numpy,
    gather,
    ge,
    gelu,
    get_num_threads,
    gradcheck,
    gt,
    index_select,
    is_grad_enabled,
    le,
    BpeTokenizer,
    load_safetensors,
    log,
    log1p,
    log_softmax,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    logsumexp,
    lt,
    manual_seed,
    masked_fill,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mm,
    multinomial,
    mul,
    ne,
    neg,
    norm,
    ones_like,
    outer,
    pow,
    prod,
    relu,
    read_safetensors_metadata,
    rsqrt,
    save_safetensors,
    set_anomaly_detection,
    set_grad_enabled,
    set_num_threads,
    sigmoid,
    sign,
    sin,
    softmax,
    sort,
    sqrt,
    stack,
    standardize,
    std,
    sub,
    sum,
    tanh,
    tensor,
    topk,
    tril,
    triu,
    var,
    where,
    zeros_like,
)

float32 = DType.F32
float64 = DType.F64
int64 = DType.I64
bool = DType.Bool

_DEFAULT_DTYPE = DType.F32


def _shape(args):
    if len(args) == 1 and not isinstance(args[0], int):
        return [int(size) for size in args[0]]
    return [int(size) for size in args]


def empty(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.empty(_shape(shape), dtype, device)


def zeros(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.zeros(_shape(shape), dtype, device)


def ones(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.ones(_shape(shape), dtype, device)


def full(shape, value, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.full(_shape((shape,)), value, dtype, device)


def rand(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.rand(_shape(shape), dtype, device)


def randn(*shape, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.randn(_shape(shape), dtype, device)


def randint(low, high, shape, device="cpu"):
    return _C.randint(low, high, _shape((shape,)), device)


def randperm(n, device="cpu"):
    return _C.randperm(n, device)


def bernoulli(*shape, p=0.5, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.bernoulli(_shape(shape), p, dtype, device)


def arange(start, end=None, step=1, dtype=None, device="cpu"):
    if end is None:
        start, end = 0, start
    if dtype is None:
        whole = _builtins.all(isinstance(value, int) for value in (start, end, step))
        dtype = int64 if whole else _DEFAULT_DTYPE
    return _C.arange(start, end, step, dtype, device)


def linspace(start, end, steps, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.linspace(start, end, steps, dtype, device)


def eye(n, dtype=_DEFAULT_DTYPE, device="cpu"):
    return _C.eye(n, dtype, device)


class _GradMode:
    _enabled = None

    def __init__(self):
        self._previous = []

    def __enter__(self):
        self._previous.append(is_grad_enabled())
        set_grad_enabled(self._enabled)
        return self

    def __exit__(self, *_):
        set_grad_enabled(self._previous.pop())
        return False

    def __call__(self, function):
        @_functools.wraps(function)
        def wrapper(*args, **kwargs):
            with type(self)():
                return function(*args, **kwargs)

        return wrapper


class no_grad(_GradMode):
    """Disable graph recording, as a context manager or as a decorator."""

    _enabled = False


class enable_grad(_GradMode):
    """Enable graph recording, as a context manager or as a decorator."""

    _enabled = True


from . import data, nn, optim  # noqa: E402
