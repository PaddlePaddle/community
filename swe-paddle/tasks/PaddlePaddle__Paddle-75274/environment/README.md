# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `5d1846ae16a8cecad8545b83d53a56e1a0eebe73`
- Gold commit: `834096be299fe1a032989a98a98572900a760a60`
- Resource: CPU
- Python dependencies: PaddlePaddle, NumPy, pytest
- Paddle rebuild required: no

The repaired artifact is `test/legacy_test/test_normal.py`. The independent verifier imports that module with a CPU-only `op_test` shim and invokes selected helper methods with `repeat_num = 2`. This preserves the graph-mode behavior under test while avoiding the original long-running statistical loop.

## Run Order

1. Check out the Base commit.
2. Restore the exact Base blob for `test_normal.py`.
3. Apply `tests/test.patch`.
4. Run the P2P test; the existing dygraph path should pass.
5. Run the two F2P tests; both static paths should fail on Base.
6. Apply `solution/code.patch`.
7. Verify the patched target blob matches the Gold commit.
8. Run `bash tests/test.sh`; all verifier tests should pass.

## Minimal Command

```bash
bash tests/test.sh
```

## Expected Matrix

| State | P2P | Scalar F2P | Tensor F2P | Full test script |
| --- | ---: | ---: | ---: | ---: |
| Base + tests | PASS | FAIL | FAIL | FAIL |
| Base + tests + solution | PASS | PASS | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
