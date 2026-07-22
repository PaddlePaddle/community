# Extend `paddle.atan2` API compatibility

Update `paddle.atan2` so that its public behavior is consistent across supported call styles while preserving all existing valid calls.

## Requirements

- Continue to support positional calls such as `paddle.atan2(x, y)` and keyword calls using `x` and `y` in both dynamic and static execution.
- Accept `input` as an alias for `x` and `other` as an alias for `y`.
- Accept an `out` Tensor in dynamic execution and write the same result into it that would otherwise be returned normally.
- Expose equivalent Tensor methods so that `x.atan2(y)` and `x.atan2(other=y)` produce the same values as `paddle.atan2(x, y)`.
- Follow NumPy-compatible broadcasting for inputs with different shapes.
- Compute correct gradients for broadcast inputs and return each gradient with the corresponding original input shape.
- Produce `float64` output for supported integer inputs.
- Handle valid empty-Tensor inputs without changing the expected output shape or dtype.
- Match `numpy.arctan2` numerically for the supported public call forms and preserve existing behavior for calls that were already valid.
