# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #57827.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `8b1a29ba9bafc16116f97422574e85d208540332` (parent of #57827 merge `3ac5e693`)
- Gold endpoint: `3ac5e693b34eb3164fe076d489dc01bea9170843`
- Resource: CPU
- GPU required: no
- Patch type: **source build required** (C++ / YAML / infermeta / adaptor).
- Paddle install: source checkout at `base_commit`, build/install so PIR dy2static paths are available.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; PIR fused-op related cases should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild**.
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- Build strategy / ResNet style cases can be heavier than unit op tests; prefer stable nodeids when selecting F2P/P2P.
- Historical PIR stack needs era-matched source build.
