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

The first run is expected to fail the seven new `TestAssertAPI` cases because `paddle._assert` is absent. Existing tests in the file remain the P2P baseline. After applying the solution and refreshing the runtime's Python package, the full unittest file should pass.

The wrapper deliberately executes the Paddle unittest file directly:

```bash
"${PYTHON_BIN:-python}" test/legacy_test/test_api_compatibility_part2.py
```

## Compatibility Risks

- The available historical build predates the exact base by 113 commits. It can provide best-effort evidence only; Python/native or Python/generated-module incompatibilities do not invalidate exact source verification.
- A runtime that imports the unpatched installed package instead of the isolated overlay will report `paddle._assert` as missing after the solution.
- Static execution depends on the runtime's control-flow assertion support. A substantially older or newer runtime may differ from the exact base even though this task changes only Python files.
- Do not patch, reset, clean, or otherwise alter the dirty `/workspace/Paddle` checkout. Use an isolated snapshot or worktree.

## Local Verification Record

Package validation passed for artifact presence, unchanged `proposal.md`, shell syntax, patch whitespace, exact file boundaries, test-first then solution application, byte-for-byte equality with the gold revision, and Python syntax compilation. Neither the wrapper nor the environment commands invokes an alternate test runner.

Runtime validation used an isolated overlay backed by the loadable Python 3.10 build at `56be465924264e1251cf127dbff56d17a7554d01`, 91 commits behind the exact base. Before the solution, all seven `TestAssertAPI` cases errored because `paddle._assert` was absent. After the solution, all seven target cases passed, including static execution. The full direct-Python wrapper ran 80 tests but reported 23 errors in unrelated compatibility APIs whose required changes postdate the older runtime, so this run is not claimed as complete P2P validation. The Python 3.9 build at `555b4a95615a35b301f348e081e56435a6d75da6` remains unusable because loading its native extension raises `SIGBUS`. The dirty `/workspace/Paddle` checkout was not modified.
