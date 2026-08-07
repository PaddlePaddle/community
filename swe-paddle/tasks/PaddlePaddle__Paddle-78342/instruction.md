# Add the public `paddle._assert` API

Implement a public assertion API callable as `paddle._assert(condition, message="")`. Its observable calling convention and behavior must be compatible with `torch._assert` while supporting Paddle dynamic and static execution.

## Calling Convention

The API must accept:

- positional arguments, such as `paddle._assert(condition, "message")`;
- keyword arguments, such as `paddle._assert(condition=True, message="message")`;
- a positional condition with a keyword message;
- an omitted message, which defaults to the empty string.

## Dynamic Behavior

In dynamic mode:

- a truthy Python condition returns normally;
- a falsy Python condition raises `AssertionError`;
- a truthy Tensor condition returns normally;
- a falsy Tensor condition raises `AssertionError`;
- the raised exception contains the supplied message exactly;
- when the message is omitted, the raised exception message is empty.

## Static Behavior

In static mode, a Tensor condition must add an executable assertion to the program. A program with a true condition must execute successfully. The condition must not be reduced to a Python boolean while the static program is being built.

## Regression Requirements

The API must be exposed from the top-level `paddle` namespace. Existing dynamic-mode, static-graph, and API compatibility behavior covered by the surrounding regression suite must remain unchanged.
