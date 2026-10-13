# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #57741.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `f984ed1a56960aeee0059c67b965406984565356` (parent of #57741 merge `4288e25e`)
- Gold endpoint: `4288e25e07895e2fd9985b7a2ec94baedac39159`
- Resource: CPU
- GPU required: no (CPU static-graph PIR memcpy translation is the F2P gate; GPU dy2static cases optional if CUDA is available)
- Patch type: **source build required**. Production changes include PIR op YAML, compat, and `pd_op_to_kernel_pass` C++.
- Paddle install: source checkout at `base_commit`, build/install so PIR dy2static paths are available.

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`. P2P CPU dy2static case should pass; F2P static PIR memcpy translation should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild** (YAML / C++ / pass changes).
5. Run `bash tests/test.sh` again; all target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

`tests/test.sh` exports `FLAGS_enable_new_ir_in_executor=1` before invoking pytest, so every selected node runs with the PIR executor enabled. Run it from the root of the Paddle source checkout; do not replace it with a bare `pytest` call that drops this setting.

Expected results on a CPU-only build: 2 passed / 0 failed / 0 errors / 3 skipped (the three CUDA dy2static nodes are skipped via `@unittest.skipIf`; skips are not counted as passes).

## Known Risks

- Historical PIR stack requires era-matched source build; newer trees may not apply patches cleanly.
- GPU memcpy cases need a CUDA build; CPU-only environments should rely on the CPU test file.
