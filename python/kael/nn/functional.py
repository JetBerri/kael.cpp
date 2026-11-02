import kael


def _reduce(losses, reduction):
    if reduction == "mean":
        return kael.mean(losses)
    if reduction == "sum":
        return kael.sum(losses)
    if reduction == "none":
        return losses
    raise ValueError(f"unknown reduction '{reduction}'")


def relu(input):
    return kael.relu(input)


def gelu(input, approximate=False):
    return kael.gelu(input, approximate)


def sigmoid(input):
    return kael.sigmoid(input)


def tanh(input):
    return kael.tanh(input)


def softmax(input, dim):
    return kael.softmax(input, dim)


def log_softmax(input, dim):
    return kael.log_softmax(input, dim)


def dropout(input, p=0.5, training=True):
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    if not training or p == 0.0:
        return input
    keep = 1.0 - p
    mask = kael.bernoulli(input.shape, p=keep, dtype=input.dtype, device=input.device)
    return input * mask / keep


def mse_loss(prediction, target, reduction="mean"):
    difference = prediction - target
    return _reduce(difference * difference, reduction)


def nll_loss(log_probabilities, targets, reduction="mean"):
    if log_probabilities.ndim != 2:
        raise ValueError(
            f"nll_loss needs log probabilities of shape [N, C], got {log_probabilities.shape}"
        )
    if targets.ndim != 1 or targets.shape[0] != log_probabilities.shape[0]:
        raise ValueError(
            f"nll_loss needs targets of shape [{log_probabilities.shape[0]}], got {targets.shape}"
        )
    if targets.dtype != kael.int64:
        raise TypeError(f"class targets must have dtype int64, got {targets.dtype}")
    picked = kael.gather(log_probabilities, 1, targets.unsqueeze(1)).squeeze(1)
    return _reduce(-picked, reduction)


def cross_entropy(logits, targets, reduction="mean"):
    return _reduce(kael.cross_entropy_losses(logits, targets), reduction)


def binary_cross_entropy_with_logits(logits, targets, reduction="mean"):
    losses = kael.relu(logits) - logits * targets + kael.log1p(kael.exp(-kael.abs(logits)))
    return _reduce(losses, reduction)
