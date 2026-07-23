# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `bbc3fbcf1b93bb5fc2f6425ccbcda22816b7c8ab`
- Resource: CPU
- GPU required: no
- Network or external model required: no
- Test framework: `pytest`
- Build path: Paddle source checkout at the base commit. Because the gold patch changes C++ core code, a source build or equivalent incremental rebuild is required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build or prepare a Python environment that loads Paddle binaries produced from that checkout.
3. Apply `tests/test.patch`.
4. Run `bash tests/test.sh`; the P2P test should pass, while the target F2P test should fail or terminate the test process.
5. Apply `solution/code.patch`.
6. Rebuild the affected Paddle binary and reinstall or otherwise expose the rebuilt package to the test environment.
7. Run `bash tests/test.sh` again; both target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The script intentionally contains only the target pytest command. Clone, checkout, patch application, build, installation, and artifact selection belong to the verifier/environment layer.

## Expected Results

- **Base + test patch**: `test_simple_pylayer_multiple_output` passes; `test_pylayer_with_partial_grad` fails or exits abnormally.
- **Base + test patch + gold patch + rebuild**: both node IDs pass.

Platform-specific source-build workarounds are not part of this benchmark task and should not be included in the task patches.
