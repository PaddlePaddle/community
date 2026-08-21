# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #59348.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `1001b3234973fb1fd2d6ede7afe918c82c792d66` (parent of #59348 merge `669a3007`)
- Gold endpoint: `669a3007e45b0b9f4600faa0a0ee3ff51fe90af3`
- Resource: CPU
- GPU required: no (CPU kernel path is the primary gate; GPU kernel is in the gold patch)
- Patch type: **source build required** (YAML / infermeta / CPU+GPU kernels).
- Paddle install: source checkout at `base_commit`, build/install so PIR op tests are available.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; PIR `sequence_mask` cases should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild**.
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- GPU kernel changes need CUDA to fully exercise; CPU-only environments should run the CPU PIR coverage path.
- Dy2static utils 白名单调整可能扩大相邻用例面，主验收以 `sequence_mask` PIR 覆盖为准。
