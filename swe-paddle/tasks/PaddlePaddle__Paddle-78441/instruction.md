# Add an `aminmax` reduction API

## Problem

Paddle provides separate minimum and maximum reductions but lacks a public operation that computes and returns both results with one API call. Add this capability as `paddle.aminmax` and as an equivalent Tensor method, with consistent behavior in dynamic and static execution.

## Public API

The operation accepts an input Tensor, an optional reduction axis, and an optional `keepdim` flag, and returns a two-element tuple `(minimum, maximum)`.

Required argument behavior:

- The input supports `float32`, `float64`, `int32`, and `int64`.
- `axis=None` reduces all dimensions.
- `axis` accepts an integer, list, or tuple, including multiple and negative axes.
- `keepdim=False` removes reduced dimensions; `keepdim=True` retains them with size one.
- `input` is accepted as an alias for the input Tensor.
- `dim` is accepted as an alias for `axis`.
- The functional API supports `out=(minimum_out, maximum_out)` and writes each result to the corresponding Tensor.
- The Tensor method and functional API produce equivalent results.

Both outputs must have the input dtype and independently match the results of applying minimum and maximum reductions with the same axis and `keepdim` arguments.

## Shapes and execution modes

Support dynamic graph, static graph, PIR, and symbolic/dynamic input shapes. Infer both output shapes correctly for:

- reduction over all dimensions;
- one or several axes;
- negative axes;
- retained dimensions;
- scalar inputs;
- empty tensors where the corresponding reduction is defined.

Both outputs always have identical shapes for the same invocation.

## Gradients

Floating-point inputs must support backward propagation from either or both outputs. The input gradient is the sum of the minimum-output and maximum-output contributions. If a reduced region contains repeated minima or repeated maxima, distribute the corresponding output gradient evenly among all equal extrema. Handle axes and `keepdim` consistently in forward and backward execution, including empty inputs.

## Compatibility and regression requirements

- Preserve existing minimum and maximum reduction behavior.
- Make the operation available through the normal Paddle API and Tensor method dispatch in rebuilt dynamic and static runtimes.
- Reject invalid arguments through Paddle's normal validation paths; do not silently alter axis or output semantics.
- Do not satisfy the task by implementing only a Python-level workaround, deleting tests, weakening assertions, or bypassing gradient and shape validation.

## Verification

The implementation must pass the focused operator/API suite and the `aminmax` symbolic-shape case supplied with this task. Verification must cover numerical outputs, gradients, aliases, `out`, static and dynamic execution, scalar and empty inputs, dynamic shapes, and symbolic shape inference. Because this feature requires compiled operator registration and generated bindings, validate it using a Paddle source build after applying the implementation patch.
