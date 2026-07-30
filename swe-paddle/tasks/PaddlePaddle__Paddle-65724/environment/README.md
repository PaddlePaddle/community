# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `9a4caad68bca019e85847eb99da57f060e01caa5`
- Gold commit: `b7c8439cd489e26c09c1db8b929285a96c64e3ed`
- Resource: CPU
- Python dependencies: PaddlePaddle, NumPy, pytest
- Paddle rebuild required: no

The verifier reads `python/paddle/io/dataloader/dataloader_iter.py` from the source checkout and loads the historical iterator classes through Python AST. Runtime dependencies come from the installed Paddle wheel.

This design validates the exact Base and Gold Python implementation while avoiding a flaky end-to-end race that depends on OS process scheduling.

## Run Order

1. Check out the Base commit.
2. Restore the exact Base blob for `python/paddle/io/dataloader/dataloader_iter.py`.
3. Apply `tests/test.patch`.
4. Run the FIFO P2P; it should pass.
5. Run the thread handoff and reset-channel F2P tests; both should fail on Base.
6. Apply `solution/code.patch`.
7. Verify the target file blob matches the Gold commit.
8. Run `bash tests/test.sh`; all verifier tests should pass.

## Minimal Command

```bash
bash tests/test.sh
```

## Expected Matrix

| State | FIFO P2P | Thread handoff F2P | Reset channel F2P | Full script |
| --- | ---: | ---: | ---: | ---: |
| Base + tests | PASS | FAIL | FAIL | FAIL |
| Base + tests + solution | PASS | PASS | PASS | PASS |

No GPU, model training, external service, or dataset download is required.
