# Add mapping-style operations to `ParameterDict`

Extend `paddle.nn.ParameterDict` with public `pop()`, `keys()`, and `values()` operations while preserving its existing parameter-container behavior.

## `pop()` Behavior

- `pop(key)` removes the parameter stored under `key` and returns that same Parameter object.
- The container length decreases after a successful removal.
- Repeated calls can remove every entry and leave an empty container.
- Removing an unknown key raises `KeyError`.
- Removed parameters must no longer appear through the container's keys or values.

## `keys()` Behavior

- `keys()` exposes every current key exactly once.
- Keys preserve the ParameterDict's insertion order.
- Parameters appended through `update()` appear after existing entries in update order.
- The returned keys stay consistent with the container length and removals.

## `values()` Behavior

- `values()` exposes every current value exactly once and in the order corresponding to `keys()`.
- Every returned value is a Paddle Parameter.
- Parameter shapes and identities are preserved.
- The number of values equals the current container length.
- Values removed through `pop()` no longer appear.

## Regression Requirements

Existing indexing, iteration, update, forward and backward execution, parameter registration, serialization, state-dict keys, state-dict loading, and output equivalence after loading must continue to work.
