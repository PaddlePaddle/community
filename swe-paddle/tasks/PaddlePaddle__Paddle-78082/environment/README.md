# Environment Notes

## Exact Revisions

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ae907b878e91dbabf3582da99f8b05a46b588fc2`
- Gold commit: `a2e4e5062dacbfef63cf4b08981b74b72ad21214`
- Gold parent: `ae907b878e91dbabf3582da99f8b05a46b588fc2`
- Loadable local runtime commit: `fa323f323bb35359c9d4ba77763834fee82a87b4` (131 commits after the exact base)
- Resource scope: CPU; no GPU is required.

## Paddle Runtime Requirements

The change is Python-only. Preferred verification uses a Paddle build produced from the exact base revision, with the patched Python package installed or copied into the build output.

A Python overlay is also valid when it starts from a compatible Paddle package and replaces the production file from the base snapshot before the test-only run, then replaces it with the solution version for the post-solution run. The overlay must retain the runtime's compiled extension, generated modules, and shared libraries while importing the overlaid Python package.

No C++, CUDA, kernel, infermeta, or native binding changes are present, so applying the solution does not require recompiling the native extension. A stale Python package in an existing build directory must still be refreshed after applying the solution patch.

## Patch And Test Order

Run from the root of a clean Paddle checkout at the exact base commit:

```bash
git apply /path/to/PaddlePaddle__Paddle-78082/tests/test.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78082/tests/test.sh
git apply /path/to/PaddlePaddle__Paddle-78082/solution/code.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-78082/tests/test.sh
```

The first run is expected to fail the nine new `TestParameterDictPopKeysValues` cases because the methods are absent. Existing container tests and the three state-dict roundtrip cases provide the P2P baseline. After applying the solution and refreshing the runtime's Python package, the full unittest file should pass.

The wrapper deliberately executes the legacy unittest file directly:

```bash
"${PYTHON_BIN:-python}" test/legacy_test/test_imperative_container_parameterdict.py
```

## Compatibility Risks

- The available loadable runtime is 131 commits newer than the exact base. It provides useful best-effort coverage but is not an exact historical P2P environment.
- A runtime that imports an installed package instead of the isolated overlay can make the pre-solution cases pass falsely because newer versions may already contain these methods.
- Layer parameter registration and state-dict behavior are sensitive to container internals. Verify both returned values and serialization tests after applying the solution.
- Do not patch, reset, clean, or otherwise alter the dirty `/workspace/Paddle` checkout. Use an isolated snapshot or worktree.

## Local Verification Record

Package validation passed for artifact presence, unchanged `proposal.md`, shell syntax, patch whitespace, exact file boundaries, test-first then solution application, byte-for-byte equality with the gold revision, and Python syntax compilation. Neither the wrapper nor the environment commands invokes an alternate test runner.

Runtime validation used isolated overlays backed by the loadable runtime at `fa323f323bb35359c9d4ba77763834fee82a87b4`, 131 commits after the exact base. Before the solution, all nine `TestParameterDictPopKeysValues` cases failed because the methods were absent. After the solution, all nine target cases passed, and the complete direct-Python file ran 32 tests with `OK`. The newer runtime was used only as a best-effort compiled environment; the Python production file was replaced with the exact base or gold version for each phase. The dirty `/workspace/Paddle` checkout was not modified.
