# Unify pinned-memory place behavior for creation APIs

## Problem

Paddle's dynamic tensor-creation and random-generation APIs do not handle `pin_memory=True` consistently. In particular, calls such as `randint(..., device="cpu", pin_memory=True)` can reject CPU placement even when the build provides a CUDA or XPU pinned-host allocator, while related APIs contain differing conversion behavior.

Implement consistent pinned-memory place semantics across the affected creation paths without changing normal behavior when pinned memory is not requested.

## Required behavior

When `pin_memory=True` is active on a supported dynamic execution path:

- An already CUDA-pinned or XPU-pinned place remains unchanged.
- A CUDA place resolves to CUDA pinned host memory.
- An XPU place resolves to XPU pinned host memory.
- A CPU place resolves to an available pinned host allocator:
  - use XPU pinned memory in an XPU-enabled build;
  - otherwise use CUDA pinned memory in a CUDA-enabled build;
  - if neither allocator is compiled in, raise `RuntimeError` with a clear message containing `Pinning memory is not supported`.
- Unsupported place kinds raise the same explicit `RuntimeError` rather than silently falling back.

The CPU behavior applies whether the device is supplied as a string or as a Paddle place object. Preserve existing device defaults, shapes, dtypes, gradient flags, output semantics, and static/PIR behavior outside the dynamic pinned-memory path.

## Affected API families

Apply these semantics consistently to:

- tensor conversion/creation, including `tensor`;
- shape- and value-based creators such as `full`, `full_like`, `eye`, `arange`, `empty`, and `empty_like`;
- random creators, including `rand`, `randn`, `randint`, and `randperm`;
- audio window creation paths that expose the same `device` and `pin_memory` behavior.

`randint` must support CPU placement with `pin_memory=True` on CUDA- or XPU-enabled builds and return a tensor with the requested shape on the corresponding pinned place.

## Verification

The implementation must pass the target tests in:

- `test/legacy_test/test_to_pinned_place.py`
- `test/legacy_test/test_eager_tensor.py`
- `test/legacy_test/test_rand.py`
- `test/legacy_test/test_randint_op.py`
- `test/legacy_test/test_randperm_op.py`

Verify both successful allocator-backed branches and explicit unsupported behavior. Accelerator tests may remain conditional on compile-time support, but a CPU-only run must not be represented as complete CUDA/XPU validation. Do not delete tests, weaken assertions, or bypass unsupported-device checks.
