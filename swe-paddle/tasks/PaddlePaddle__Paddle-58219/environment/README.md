# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `4dbd3f7d8a3a54066939a2e1acc46fadadf65c11`
- Resource: CPU
- GPU required: no
- Network service or external model required: no
- Suggested Python: 3.10
- Test framework: Python `unittest`
- Patch type: pure Python; no C++/CUDA/kernel rebuild is required once a compatible Paddle binary is available and the checkout's Python sources are loaded.

## Verified Compatibility Environment

The Run/Test/Fix behavior was reproduced with:

- Image: `paddlepaddle/paddle:2.6.0`
- Container platform: `linux/amd64`
- Python: `3.10.13`
- Paddle binary commit: `e032331bf78b0f9b51806c6761254c8b977f02b4`

The image contains Python-layer PIR renames made after this task's base commit. The
verification therefore used a runtime-only compatibility mapping between the old
and new monkey-patch entrypoint names while loading the PR's before/after target
logic. That mapping is environment glue and is not part of either task patch.
An exact era-matched source build remains the preferred verifier environment.

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Prepare a compatible compiled Paddle runtime and ensure the checkout's
   `python/` sources are the package under test.
3. Apply `tests/test.patch`.
4. Run `bash tests/test.sh`; the four new operator tests should error, while the
   four existing regression tests should pass.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh` again; all eight tests should pass. The gold patch is
   Python-only, so no native rebuild is needed.

## Minimal Test Command

```bash
python test/legacy_test/test_math_op_patch_pir.py -v
```

The script intentionally contains only the target test command. Checkout, patch
application, binary selection, and Python source overlay belong to the verifier.

## Expected Results

- **Base + test patch**: `test_item`, `test_place`, `test_some_dim`, and
  `test_math_exists` pass; `test_pow`, `test_floordiv`, `test_mod`, and
  `test_matmul` error with unsupported-operator `TypeError`.
- **Base + test patch + gold patch**: all eight tests pass.

## Patch Provenance and Risks

- GitHub records the PR base and head on non-linear histories. Do not regenerate
  the patch with `git diff base_commit..head`, which includes unrelated develop
  changes. The provided patches contain only the three files in GitHub's PR diff.
- No exact wheel for the 2023 base commit is available in the current nightly
  index; PIR Python/binary compatibility must be fixed by the verifier environment.
- The original tests explicitly select CPU and use `assert_allclose`. Random
  inputs have no fixed seed, but do not depend on a probabilistic threshold.
- Scalar reverse power is outside this task because the original PR did not keep
  that test enabled.
