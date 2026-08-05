# Environment Notes

SWE-Paddle task for PaddlePaddle/Paddle PR #77495.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `0604f65af5397848b6803c2bf577b9b82b8d8e08`
- Minimum verifier resource: CPU
- Optional coverage: one CUDA GPU
- XPU: non-default dilation is outside this task's implementation scope
- Build path: source build required

The patch changes operator YAML, InferMeta, C++ CPU kernels, CUDA kernels, and
Python APIs. A prebuilt wheel or Python overlay cannot validate the task.
Rebuild Paddle after applying the gold patch so generated operator bindings and
native kernels are refreshed.

## Run / Test / Fix

1. Check out Paddle at the exact base commit in a clean worktree.
2. Apply `tests/test.patch` and complete an era-matched source build.
3. Run `bash "$TASK_DIR/tests/test.sh"`. The two P2P nodes must pass; F2P nodes
   must fail only because dilation or the target call form is unsupported.
4. Return to a clean checkout of the same base and apply `tests/test.patch`
   followed by `solution/code.patch`.
5. Reconfigure and rebuild Paddle so generated bindings and native kernels are
   refreshed.
6. Run `bash "$TASK_DIR/tests/test.sh"` again; every selected node must pass.

## Minimal Test Command

```bash
TASK_DIR=/path/to/community/swe-paddle/tasks/PaddlePaddle__Paddle-77495
bash "$TASK_DIR/tests/test.sh"
```

The independent verifier uses only CPU tensors. CUDA coverage may additionally
run the same API cases on a CUDA place and the upstream max-pool dilation tests.
`PYTHON_BIN` may select the interpreter importing the exact source build.

## F2P / P2P Classification

- P2P: `test_existing_positional_calls_1d/2d/3d`
- F2P feature: functional and layer forward cases, exact backward gradients,
  indexed 2D/3D OpTest forward/gradient nodes, asymmetric dilation, static
  graph, mask, `ceil_mode`, and layer repr
- F2P compatibility: `test_compatibility_trap_1d/2d/3d`

Before the gold patch, compatibility nodes must fail with an unsupported call,
wrong return contract, or wrong dilated output. A crash, import error, or
linkage error is an invalid environment rather than an expected F2P result.

## Known Risks

- A source rebuild is mandatory after the gold patch.
- `max_pool2d_with_index` and `max_pool3d_with_index` gain a dilation input.
  The focused upstream OpTest wrappers in `tests/test.patch` must be applied;
  stale wrappers indicate an invalid test state, not a kernel regression.
- CUDA and DCU compilation paths are touched even though the minimum verifier
  is CPU-only.
- The original PR head merged contemporaneous `develop`; the supplied gold
  patch deliberately excludes those unrelated changes.
- The original author environment is recorded in the proposal, but the minimum
  Linux CPU build configuration and build duration have not been confirmed.