import math

import kael

from . import functional as F
from . import init


class Parameter(kael.Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data.detach())
        self.requires_grad = requires_grad


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if "_parameters" not in self.__dict__:
            raise AttributeError("call super().__init__() before assigning attributes")
        for table in (self._parameters, self._modules, self._buffers):
            table.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name, value):
        self._buffers[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not define forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix=""):
        seen = set()
        for name, parameter in self._walk_parameters(prefix):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter

    def _walk_parameters(self, prefix):
        for name, parameter in self._parameters.items():
            yield prefix + name, parameter
        for name, child in self._modules.items():
            yield from child._walk_parameters(prefix + name + ".")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def modules(self):
        yield self
        for child in self._modules.values():
            yield from child.modules()

    def train(self, training=True):
        for module in self.modules():
            object.__setattr__(module, "training", training)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device):
        with kael.no_grad():
            for parameter in self.parameters():
                parameter._move_to_(device)
                parameter._set_grad_device(device)
            for module in self.modules():
                for name, buffer in module._buffers.items():
                    module.register_buffer(name, buffer.to(device))
        return self

    def state_dict(self):
        state = {
            name: parameter.detach() for name, parameter in self._walk_parameters("")
        }
        for prefix, module in self._named_modules(""):
            for name, buffer in module._buffers.items():
                state[prefix + name] = buffer
        return state

    def _named_modules(self, prefix):
        yield prefix, self
        for name, child in self._modules.items():
            yield from child._named_modules(prefix + name + ".")

    def load_state_dict(self, state):
        targets = dict(self._walk_parameters(""))
        for prefix, module in self._named_modules(""):
            for name, buffer in module._buffers.items():
                targets[prefix + name] = buffer
        missing = [name for name in targets if name not in state]
        if missing:
            raise KeyError(f"state dict is missing {missing}")
        with kael.no_grad():
            for name, target in targets.items():
                if state[name].shape != target.shape:
                    raise ValueError(
                        f"{name} has shape {target.shape} "
                        f"but the state dict has {state[name].shape}"
                    )
                target.copy_(state[name])


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=kael.float32):
        super().__init__()
        self.weight = Parameter(kael.empty(out_features, in_features, dtype=dtype))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = Parameter(kael.empty(out_features, dtype=dtype))
            limit = 1 / math.sqrt(in_features) if in_features > 0 else 0
            init.uniform_(self.bias, -limit, limit)

    def forward(self, input):
        output = kael.matmul(input, self.weight.mT)
        return output if self.bias is None else output + self.bias


class ReLU(Module):
    def forward(self, input):
        return kael.relu(input)


class GELU(Module):
    def __init__(self, approximate=False):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return kael.gelu(input, self.approximate)


class Sigmoid(Module):
    def forward(self, input):
        return kael.sigmoid(input)


class Tanh(Module):
    def forward(self, input):
        return kael.tanh(input)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return kael.softmax(input, self.dim)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        return F.dropout(input, self.p, self.training)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, dtype=kael.float32):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(kael.ones(self.normalized_shape, dtype=dtype))
        self.bias = Parameter(kael.zeros(self.normalized_shape, dtype=dtype))

    def forward(self, input):
        count = len(self.normalized_shape)
        if input.shape[-count:] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm over {self.normalized_shape} got input shape {input.shape}"
            )
        return kael.standardize(input, count, self.eps) * self.weight + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=kael.float32):
        super().__init__()
        self.weight = Parameter(kael.empty(num_embeddings, embedding_dim, dtype=dtype))
        init.normal_(self.weight, 0.0, 1.0)

    def forward(self, indices):
        selected = kael.index_select(self.weight, 0, indices.flatten())
        return selected.reshape(*indices.shape, self.weight.shape[1])


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for position, layer in enumerate(layers):
            setattr(self, str(position), layer)

    def forward(self, input):
        for layer in self._modules.values():
            input = layer(input)
        return input


class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for position, module in enumerate(modules):
            setattr(self, str(position), module)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, position):
        return self._modules[str(position)]


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, self.reduction)


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        return F.mse_loss(prediction, target, self.reduction)
