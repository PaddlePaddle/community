# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `fe811625db37300f74064a52e80c130d7ae347ed`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. A full Paddle build is not required; the verifier compiles a lightweight C++17 harness containing the checked-out scope-selection logic.

The harness provides controlled `Scope` and scope-variable doubles. It models child creation, repeated execution, legacy executor behavior, and a stale cached child whose parent now contains a different valid child. The compiled executable is cached by source hash.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the target behavior should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; the target behavior should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.
