import math

import kael


def _fans(tensor):
    if tensor.ndim < 2:
        raise ValueError(f"fans need a tensor with at least 2 dimensions, got {tensor.shape}")
    receptive_field = tensor.numel() // (tensor.shape[0] * tensor.shape[1])
    return tensor.shape[1] * receptive_field, tensor.shape[0] * receptive_field


@kael.no_grad()
def uniform_(tensor, low=0.0, high=1.0):
    return tensor.copy_(kael.rand(tensor.shape, dtype=tensor.dtype) * (high - low) + low)


@kael.no_grad()
def normal_(tensor, mean=0.0, std=1.0):
    return tensor.copy_(kael.randn(tensor.shape, dtype=tensor.dtype) * std + mean)


@kael.no_grad()
def zeros_(tensor):
    return tensor.zero_()


@kael.no_grad()
def ones_(tensor):
    return tensor.fill_(1)


def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _fans(tensor)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -limit, limit)


def kaiming_uniform_(tensor, a=0.0):
    fan_in, _ = _fans(tensor)
    limit = math.sqrt(2.0 / (1.0 + a * a)) * math.sqrt(3.0 / fan_in)
    return uniform_(tensor, -limit, limit)


def kaiming_normal_(tensor, a=0.0):
    fan_in, _ = _fans(tensor)
    return normal_(tensor, 0.0, math.sqrt(2.0 / (1.0 + a * a)) / math.sqrt(fan_in))
