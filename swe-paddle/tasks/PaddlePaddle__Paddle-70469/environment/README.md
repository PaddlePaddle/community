# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `052874d0c95fe9bcae0c3e0ac60d857b238e6d70`
- Gold commit: `5cc69a1b9adbb4b3f3ce30312c0b031be503dff4`
- Resource: CPU
- Python dependencies: pytest
- Paddle rebuild required: no

The verifier extracts the target functions and module state from the checked-out `fused_dropout_add.py` through Python AST. It executes them with controlled Paddle and `_C_ops` doubles.

The doubles make the reference `dropout + add` result deterministic and record whether the fused operator was invoked. This validates control flow and API semantics without requiring a GPU or executing the affected fused kernel.

## Run Order

1. Check out the Base commit.
2. Restore the exact Base blob for `fused_dropout_add.py`.
3. Apply `tests/test.patch`.
4. Run the `p == 0` P2P; it should pass.
5. Run the training-fallback and warning-once F2P tests; both should fail on Base.
6. Apply `solution/code.patch`.
7. Verify the target file blob matches the Gold commit.
8. Run `bash tests/test.sh`; all verifier tests should pass.

## Minimal Command

```bash
bash tests/test.sh
```

## Expected Matrix

| State | `p == 0` P2P | Training fallback F2P | Warning/inference F2P | Full script |
| --- | ---: | ---: | ---: | ---: |
| Base + tests | PASS | FAIL | FAIL | FAIL |
| Base + tests + solution | PASS | PASS | PASS | PASS |

No GPU, CUDA runtime, fused kernel execution, external service, or dataset is required.
