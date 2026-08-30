# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #59021.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `a53f40972d9dea85b44e6eae288f14c1bd01e3a7` (parent of #59021 merge `3af9eb7e`)
- Gold endpoint: `3af9eb7eb21f80e81f3573c427feeebbd621a72a`
- Resource: CPU
- GPU required: no
- Patch type: **source build required** (C++ adaptor / utils + YAML).
- Paddle install: source checkout at `base_commit`, build/install so PIR executor paths are available.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; SelectedRows `len` / PIR coverage cases should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild**.
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- `test_len` 本身未改文件；F2P 依赖 PIR / Dy2St 参数化路径与 executor flag。
- `test_fuse_elewise_add_act_pass` 在 Apple / 无 GPU 环境下可能有额外跳过逻辑，以 CMake 门禁为准。
