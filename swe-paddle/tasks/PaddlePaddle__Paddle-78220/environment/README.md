# Environment Notes

## Exact Revisions

- Repository: `PaddlePaddle/Paddle`
- Base commit: `56be465924264e1251cf127dbff56d17a7554d01`
- Gold commit: `bfe91230d558176d2d932b50953cb7b4391065d1`
- Gold parent: `56be465924264e1251cf127dbff56d17a7554d01`
- Available local build revision: `555b4a95615a35b301f348e081e56435a6d75da6` (22 commits behind the base)
- Resource scope: CPU; no GPU is required.

## Paddle Runtime Requirements

The patches are Python-only, so either of these environments is suitable:

1. A Paddle source build produced from the exact base revision, with the patched source tree's `python` package used at runtime.
2. A Python overlay of the patched source tree on a compatible Paddle build whose compiled extension exposes the `log_softmax` binding used by the base revision, including dtype casting and the `out` argument.

The overlay must load Python modules from the patched checkout while loading Paddle's compiled extension and generated runtime modules from the matching build output. Confirm the loaded package and extension locations before interpreting test results. A wheel or build from an arbitrary newer revision is not an exact historical verification environment.

## Patch And Test Order

Run from the root of a clean Paddle checkout at the exact base commit:

```bash
git apply /path/to/PaddlePaddle__Paddle-78220/tests/test.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78220/tests/test.sh
git apply /path/to/PaddlePaddle__Paddle-78220/solution/code.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78220/tests/test.sh
```

The first test run is expected to expose the missing public routes and parameter semantics. After the solution patch, both unittest files should pass.

The wrapper deliberately executes each unittest file directly:

```bash
"${PYTHON_BIN:-python}" test/legacy_test/test_compat_log_softmax.py
"${PYTHON_BIN:-python}" test/legacy_test/test_log_softmax.py
```

Direct execution is required because `test_log_softmax.py` enables static mode in its `__main__` block before starting `unittest`.

## Compatibility Risks

- The available local build predates the exact base by 22 commits. It can provide useful best-effort coverage, but a failure caused by a Python/compiled-extension signature mismatch is not evidence that the exact base-to-gold patch is incorrect.
- Static and PIR tests depend on build-time capabilities and generated bindings. Byte-equivalent source verification does not replace execution against an exact base build.
- An overlay that accidentally imports Python modules from the installed build instead of the patched snapshot can produce false failures for missing APIs. Conversely, importing an incompatible extension can fail before the target tests run.
- Do not patch, reset, clean, or otherwise alter the dirty `/workspace/Paddle` checkout to assemble the verification environment. Use an isolated snapshot.

## Local Verification Record

The package checks passed for artifact presence, proposal content, shell syntax, patch whitespace, exact file boundaries, test-first then solution patch application, byte-for-byte equality with the gold revision, and Python syntax compilation. The later proposal edit only clarifies the rank-dependent default-dimension behavior; it does not change the production or test patches. Neither `tests/test.sh` nor the environment commands invokes an alternate test runner.

Runtime execution was attempted in two isolated overlays. The Python 3.9 runtime associated with the available ancestor build (`555b4a95615a35b301f348e081e56435a6d75da6`) raised `SIGBUS` while loading `libpaddle.so`, including when loaded directly, so the target wrapper could not start there. A loadable Python 3.10 Paddle package from commit `826269a123ffb4a2213aadc04bbfa9c601eb0fc6` was also tested only as a non-exact fallback; the wrapper stopped during import because that newer package lacks `ForbidKeywordsIgnoreOneParamDecorator`, which the gold-era Python patch requires. No target test results are claimed from either incompatible runtime, and `/workspace/Paddle` was not modified.
