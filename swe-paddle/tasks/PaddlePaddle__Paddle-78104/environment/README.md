# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `d85ad0fca9513ff7d1f0a552649f9136e94cf2a5`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. No Paddle rebuild or wheel installation is required; the verifier executes the checked-out Python function through AST.

The verifier supplies controlled `core.Place`, `CPUPlace`, `CUDAPlace`, `XPUPlace`, and `CustomPlace` doubles. It validates device type and device-id preservation without requiring accelerator hardware.

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
