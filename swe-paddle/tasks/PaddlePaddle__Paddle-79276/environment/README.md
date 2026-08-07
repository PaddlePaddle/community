# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `8cacdfd15bc89296682c784df5b1685a7ca6e408`
- Gold/head commit: `fa6bfed3dde252b97c9db8e32ce4d8bdd813b8a4`
- Resource: CPU
- GPU required: no
- Network or external model required: no
- Test framework: `pytest`
- Build path: Paddle source checkout at the base commit. The gold patch changes C++ infermeta, so real verification requires a source build or an equivalent compiled runtime that includes the patched C++ code.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build or prepare a Python environment that loads Paddle binaries produced from that checkout.
3. Apply `tests/test.patch`.
4. Run `bash tests/test.sh`; the P2P guards should pass, while the target F2P test should fail on the unpatched implementation.
5. Apply `solution/code.patch`.
6. Rebuild the affected Paddle binary and reinstall or otherwise expose the rebuilt package to the test environment.
7. Run `bash tests/test.sh` again; the P2P guards and F2P target should all pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The script intentionally contains only the dedicated pytest command. Clone, checkout, patch application, build, installation, and artifact selection belong to the verifier/environment layer.

## Expected Results

- **Base + test patch**: compatible zero-size and ordinary-input P2P guards pass; the mixed zero-size/non-zero-size F2P test fails.
- **Base + test patch + gold patch + rebuild/equivalent compiled runtime**: all tests in the dedicated class pass.

Runtime Run/Test/Fix validation requires a Paddle source build or equivalent compiled runtime that includes the patched C++ infermeta code; that validation is the SWE-Paddle verifier's responsibility.

Platform-specific source-build workarounds are not part of this benchmark task and should not be included in the task patches.
