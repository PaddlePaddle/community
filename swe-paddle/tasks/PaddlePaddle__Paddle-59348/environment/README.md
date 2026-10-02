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
- 上游 CTest 通过 `tools/test_runner.py` 执行本测试，并因 `static_mode_white_list` 先调用 `paddle.enable_static()`；测试文件自身的 `__main__` 不开启静态图。`tests/test.sh` 在启动 pytest 前显式调用 `paddle.enable_static()`，否则 `TestSequenceMaskOpError::test_errors` 会在动态图下误报失败。
- `op_test` / `white_list` 需要 `test/legacy_test` 与 `test` 在 `PYTHONPATH` 中，`tests/test.sh` 已设置。
- 预期 Gold 结果：15 项全部通过（13 项 PIR `sequence_mask` 目标用例 + `TestSequenceMaskOpError`、`TestSequenceMaskWithEmptyTensor` 2 项原有行为）；skip 不计为通过。
