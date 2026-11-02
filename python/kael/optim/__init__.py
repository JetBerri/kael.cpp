import math

import kael


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        if lr < 0:
            raise ValueError(f"learning rate must be non-negative, got {lr}")
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    @kael.no_grad()
    def step(self):
        for index, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            if gradient is not None:
                self._update(index, parameter, gradient)

    def _update(self, index, parameter, gradient):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def _update(self, index, parameter, gradient):
        direction = gradient
        if self.weight_decay != 0:
            direction = direction + parameter.detach() * self.weight_decay
        if self.momentum != 0:
            velocity = self._velocity.get(index)
            if velocity is None:
                velocity = direction.clone()
            else:
                velocity.mul_(self.momentum).add_(direction)
            self._velocity[index] = velocity
            direction = velocity
        parameter.sub_(direction * self.lr)


class _AdamBase(Optimizer):
    def __init__(self, parameters, lr, betas, eps, weight_decay):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._state = {}

    def _moments(self, index, parameter):
        if index not in self._state:
            self._state[index] = [0, kael.zeros_like(parameter), kael.zeros_like(parameter)]
        return self._state[index]

    def _adam_step(self, index, parameter, gradient):
        state = self._moments(index, parameter)
        state[0] += 1
        step, first, second = state
        beta1, beta2 = self.betas
        first.mul_(beta1).add_(gradient * (1 - beta1))
        second.mul_(beta2).add_(gradient * gradient * (1 - beta2))
        corrected_first = first / (1 - beta1**step)
        corrected_second = second / (1 - beta2**step)
        parameter.sub_(corrected_first / (kael.sqrt(corrected_second) + self.eps) * self.lr)


class Adam(_AdamBase):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr, betas, eps, weight_decay)

    def _update(self, index, parameter, gradient):
        if self.weight_decay != 0:
            gradient = gradient + parameter.detach() * self.weight_decay
        self._adam_step(index, parameter, gradient)


class AdamW(_AdamBase):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps, weight_decay)

    def _update(self, index, parameter, gradient):
        parameter.mul_(1 - self.lr * self.weight_decay)
        self._adam_step(index, parameter, gradient)


@kael.no_grad()
def clip_grad_norm_(parameters, max_norm):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return 0.0
    total = math.sqrt(sum(kael.sum(g * g).item() for g in gradients))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)
    return total


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.steps = 0
        self._apply()

    def get_lr(self):
        raise NotImplementedError

    def _apply(self):
        self.optimizer.lr = self.get_lr()

    def step(self):
        self.steps += 1
        self._apply()


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        if step_size <= 0:
            raise ValueError(f"step size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.steps // self.step_size)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, total_steps, min_lr=0.0):
        if total_steps <= 0:
            raise ValueError(f"cosine schedule length must be positive, got {total_steps}")
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        progress = min(self.steps, self.total_steps) / self.total_steps
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(f"warmup of {warmup_steps} does not fit a schedule of {total_steps}")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.steps < self.warmup_steps:
            return self.base_lr * (self.steps + 1) / self.warmup_steps
        span = self.total_steps - self.warmup_steps
        progress = min(self.steps - self.warmup_steps, span) / span
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
