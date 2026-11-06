# ADR 0005: Define-by-run autograd graph

## Status

Accepted in phase 7 (tag `chapter-07-autograd`). Still in force at `v1.0.0`.

## Context

Reverse-mode differentiation needs a record of the operations that produced a value. The main designs are:

- A static graph built before execution, as in TensorFlow 1. It allows whole-graph optimization but separates the model definition from ordinary C++ or Python control flow.
- A global tape that appends every operation and replays it backwards. It is simple, but every tensor shares one tape, and freeing memory or differentiating two unrelated losses becomes awkward.
- A define-by-run graph, as in PyTorch: each result points to the node that produced it, and the graph exists only as long as some tensor references it.

kael targets readers who know PyTorch, and its models use plain loops and conditionals, so the graph has to follow normal control flow.

## Decision

kael builds the graph while operations run:

- Each `TensorImpl` holds an optional `AutogradMeta` (`include/kael/autograd/meta.h`) with `requires_grad`, `grad`, the `grad_fn` node that produced it, and a weak reference to the leaf's `AccumulateGrad` node. Tensors outside differentiation pay one null pointer.
- A `Node` holds its input edges as `shared_ptr`s and computes vector-Jacobian products: it maps the gradient of its output to the gradients of its inputs, and never forms a Jacobian.
- Operations record history only when grad mode is on and an input requires gradients. Grad mode is a thread-local flag set by `NoGradGuard`.
- `backward` counts the dependencies of every reachable node, then processes nodes whose gradients are complete, so a value used several times receives the sum of all its contributions before it propagates.
- Saved tensors record the storage version counter, and backward fails if an in-place op changed them. In-place ops on tensors that are part of the graph are rejected instead of rewriting history.
- Gradients are first order only: backward does not record a graph of its own.

## Consequences

- Models are ordinary code. Loops, early returns and data-dependent branches need no special API.
- Memory follows references: when the last tensor of a graph is destroyed, its nodes go with it. The `Node` destructor walks long chains iteratively, so deep graphs do not overflow the stack.
- In-place operations are more restricted than in PyTorch, which keeps the version and history rules simple to explain and test.
- Higher-order derivatives, such as gradients of gradients, are out of scope.
- `gradcheck` compares every formula with finite differences, and chapter 06 derives the formulas.
