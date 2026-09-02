# Environment Notes

## Static Preparation

- Repository: `PaddlePaddle/Paddle`
- Base commit: `3cb4059b8e870c818031779af94eae728177c2ac`
- Gold/head commit: `d2427e1d53fbd4d65623af43fdaf26d76740feb3`
- PR: https://github.com/PaddlePaddle/Paddle/pull/79321
- Resource: CPU
- GPU required: no
- Distributed launch required: no
- External service required: no
- Patch type: Python-only production change plus Python legacy test patch
- Build requirement: no C++ rebuild is expected for the production patch.

The package contents can be prepared statically: apply `tests/test.patch` to the Paddle checkout at the base commit, then apply `solution/code.patch` for the gold behavior. The production change only edits `python/paddle/nn/layer/layers.py`.

## Verifier Runtime Responsibility

The verifier is responsible for providing a compatible Paddle runtime, performing runtime Run/Test/Fix, and recording the results. Runtime verification must ensure Python imports resolve to the patched checkout's `python/` tree rather than an unrelated installed Paddle source tree or wheel. For example, the verifier may need to set the repository-local import path or otherwise confirm that `paddle.nn.layer.layers` is loaded from the patched checkout before executing the nodeids.

## Run Order

1. Check out `PaddlePaddle/Paddle` at `3cb4059b8e870c818031779af94eae728177c2ac`.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the existing tuple-unpacking P2P should pass and the new named-field F2P should fail on the unmodified base.
4. Apply `solution/code.patch`.
5. Ensure imports resolve to the patched checkout.
6. Run `bash tests/test.sh` again; both target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```