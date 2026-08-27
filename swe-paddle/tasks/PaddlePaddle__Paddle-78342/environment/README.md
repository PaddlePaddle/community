# Environment Notes

## Exact Revisions

- Repository: `PaddlePaddle/Paddle`
- Base commit: `fa323f323bb35359c9d4ba77763834fee82a87b4`
- Gold commit: `f92a35feea4acf62b2df2259ae491b992851f854`
- Gold parent: `fa323f323bb35359c9d4ba77763834fee82a87b4`
- Available local Python 3.9 historical build: `555b4a95615a35b301f348e081e56435a6d75da6` (113 commits behind the base; native extension is not loadable in this container)
- Loadable local Python 3.10 build: `56be465924264e1251cf127dbff56d17a7554d01` (91 commits behind the base)
- Resource scope: CPU; no GPU is required.

## Paddle Runtime Requirements

The changes are Python-only. Preferred verification uses a Paddle build produced from the exact base revision, with the patched checkout's Python package installed or copied into the build output.

A Python overlay is also valid when it starts from a compatible Paddle package and replaces the three patched production files. The overlay must load the patched top-level and testing modules while retaining the build's compiled extension, generated modules, and shared libraries. Confirm the loaded package path and Paddle commit before interpreting results.

No C++, CUDA, kernel, infermeta, or generated binding changes are present, so the solution does not require recompiling the native extension. A stale Python package in an existing build directory must still be refreshed after applying the solution patch.

## Patch And Test Order

Run from the root of a clean Paddle checkout at the exact base commit:

```bash
git apply /path/to/PaddlePaddle__Paddle-78342/tests/test.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78342/tests/test.sh
git apply /path/to/PaddlePaddle__Paddle-78342/solution/code.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78342/tests/test.sh
```

The first run is expected to fail the seven new `TestAssertAPI` cases because `paddle._assert` is absent, while the P2P selection already passes. After applying the solution and refreshing the runtime's Python package, both selections should pass.

The wrapper runs a narrowed selection instead of the whole compatibility file:

```bash
# P2P tests (pass-to-pass)
"${PYTHON_BIN:-python}" -m pytest test/legacy_test/test_assert_close.py::TestAssertClose -q

# F2P tests (fail-to-pass)
"${PYTHON_BIN:-python}" -m pytest test/legacy_test/test_api_compatibility_part2.py::TestAssertAPI -q
```

`TestAssertClose` is the P2P guard because `assert_close` lives in `python/paddle/testing/_comparison.py` and is re-exported from `python/paddle/testing/__init__.py`, the two files the solution patch edits besides the top-level namespace. Running the full `test_api_compatibility_part2.py` is not usable as a P2P baseline: unrelated compatibility cases in that file depend on API changes that a non-exact runtime does not carry, so the file is red both before and after the solution.

## Compatibility Risks

- The available historical build predates the exact base by 113 commits. It can provide best-effort evidence only; Python/native or Python/generated-module incompatibilities do not invalidate exact source verification.
- A runtime that imports the unpatched installed package instead of the isolated overlay will report `paddle._assert` as missing after the solution.
- Static execution depends on the runtime's control-flow assertion support. A substantially older or newer runtime may differ from the exact base even though this task changes only Python files.
- Do not patch, reset, clean, or otherwise alter the dirty `/workspace/Paddle` checkout. Use an isolated snapshot or worktree.

## Local Verification Record

Package validation passed for artifact presence, unchanged `proposal.md`, shell syntax, patch whitespace, exact file boundaries, test-first then solution application, byte-for-byte equality with the gold revision, and Python syntax compilation. Neither the wrapper nor the environment commands invokes an alternate test runner.

Runtime validation used an isolated overlay backed by the loadable Python 3.10 build at `56be465924264e1251cf127dbff56d17a7554d01`, 91 commits behind the exact base. Before the solution, all seven `TestAssertAPI` cases errored because `paddle._assert` was absent. After the solution, all seven target cases passed, including static execution. The full direct-Python wrapper ran 80 tests but reported 23 errors in unrelated compatibility APIs whose required changes postdate the older runtime, so running the whole file is not usable as P2P validation. The Python 3.9 build at `555b4a95615a35b301f348e081e56435a6d75da6` remains unusable because loading its native extension raises `SIGBUS`. The dirty `/workspace/Paddle` checkout was not modified.

The narrowed wrapper was validated separately. In the fail-before state, on a base checkout with only `tests/test.patch` applied and an ancestor CPU runtime at Paddle commit `7743e779aff3e35b8bd748b2c69b9332f5d8dfd7`, `test_assert_close.py::TestAssertClose` passed 24 cases while all seven `TestAssertAPI` cases failed. Running the whole `test_api_compatibility_part2.py` in that same state reported 79 errors out of 80 cases, which is why the file itself cannot serve as a P2P baseline. The pass-after state was confirmed on a runtime built from the exact base commit, with the three patched production files installed into that package: both the P2P and the F2P selection pass.
