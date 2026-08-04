# Add compatible `log_softmax` public APIs

Implement `log_softmax` so that Paddle's standard and compatibility-facing public APIs expose consistent behavior without regressing the existing operator.

## Public access paths

The operation must be callable through all of these routes:

- `paddle.nn.functional.log_softmax`
- `paddle.log_softmax`
- `Tensor.log_softmax`
- `paddle.special.log_softmax`
- `paddle.compat.nn.functional.log_softmax`

For equivalent arguments, all five routes must produce numerically equivalent results.

## Standard Paddle API behavior

`paddle.nn.functional.log_softmax` must preserve its existing positional and keyword behavior for `x`, `axis`, `dtype`, and `name`. It must additionally:

- accept `input` as an alias for `x`;
- accept `dim` as an alias for `axis`;
- accept a keyword-only `out` Tensor and write the result into it;
- reject calls that provide both names from either alias pair (`x` with `input`, or `axis` with `dim`).

The top-level, Tensor, and special-module routes must expose the same compatible operation.

## Compatibility API behavior

`paddle.compat.nn.functional.log_softmax` must accept an input Tensor, optional `dim`, optional `dtype`, and keyword-only `out`.

When `dim` is `None`, select the default dimension from the input rank:

- rank 0, 1, or 3: dimension 0;
- every other rank: dimension 1.

The `dtype` argument must accept supported Paddle dtype objects and dtype strings. When specified, computation and output use that dtype; otherwise the input dtype is preserved. When `out` is supplied, the result must be written to and returned through that Tensor.

For compatibility with the corresponding external API, an integer `_stacklevel` keyword is accepted and ignored. Paddle-specific keywords `x`, `axis`, and `name` must be rejected, as must unsupported or misspelled keywords.

## Regression requirements

Preserve existing CPU numerical results, gradients, dtype behavior, static-graph execution, dynamic execution, and PIR execution. Existing positional calls and standard Paddle keyword calls must continue to work.
