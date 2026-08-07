# Environment Notes

## Exact Revisions

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ea0f979936ab101a91a8739bdb0a528b8df42ef7`
- Gold commit: `7c19c94684c0e93b6d5f2b288d34d2a61e39b02a`
- Gold parent: `ea0f979936ab101a91a8739bdb0a528b8df42ef7`
- Loadable local runtime commit: `ae907b878e91dbabf3582da99f8b05a46b588fc2` (363 commits after the exact base)
- Resource scope: CPU; no GPU is required.

## Paddle Runtime Requirements

The changes are Python-only. Preferred verification uses a Paddle build produced from the exact base revision, with the patched Python package installed or copied into the build output.

A Python overlay is also valid when it retains a compatible runtime's compiled extension and generated modules while replacing the two production paths with the exact base or gold source. For the pre-solution run, the new RNN utility module must be absent as it is at the exact base; leaving a newer runtime's copy importable would create a false pass. Confirm both the loaded Paddle package and production module paths before interpreting results.

No C++, CUDA, kernel, infermeta, or native binding changes are present, so applying the solution does not require recompiling the native extension. A stale Python package in an existing build directory must still be refreshed after applying the solution patch.

## Patch And Test Order

Run from the root of a clean Paddle checkout at the exact base commit:

```bash
git apply /path/to/PaddlePaddle__Paddle-77749/tests/test.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-77749/tests/test.sh
git apply /path/to/PaddlePaddle__Paddle-77749/solution/code.patch
PYTHON_BIN=python bash /path/to/PaddlePaddle__Paddle-77749/tests/test.sh
```

The first run is expected to fail while importing the absent `paddle.nn.utils.rnn` module. After applying the solution and refreshing the runtime's Python package, all 21 target cases should pass.

The wrapper deliberately executes the legacy unittest file directly:

```bash
"${PYTHON_BIN:-python}" test/legacy_test/test_rnn_utils.py
```

## Compatibility Risks

- The available loadable runtime is 363 commits newer than the exact base. It supplies best-effort native support but is not an exact historical runtime.
- A pre-solution overlay that leaves the newer runtime's RNN utility module in place invalidates F2P verification.
- These APIs rely on Tensor creation, concatenation, stacking, transposition, and slicing behavior. A substantially different runtime can introduce unrelated failures.
- The exact test patch contains only new F2P coverage, so broad P2P confidence requires existing Paddle regression suites beyond this target wrapper.
- Do not patch, reset, clean, or otherwise alter the dirty `/workspace/Paddle` checkout. Use an isolated snapshot or worktree.

## Local Verification Record

Package validation passed for artifact presence, unchanged `proposal.md`, shell syntax, patch whitespace, exact file boundaries, test-first then solution application, byte-for-byte equality with the gold revision, and Python syntax compilation. Neither the wrapper nor the environment commands invokes an alternate test runner.

Runtime validation used isolated overlays backed by the loadable runtime at `ae907b878e91dbabf3582da99f8b05a46b588fc2`, 363 commits after the exact base. The base overlay replaced the utility exports with the exact base file and removed the newer runtime's RNN utility module; the test-only run then failed during import with `ModuleNotFoundError`, as expected. The gold overlay used both exact gold production files, and the complete direct-Python wrapper ran 21 tests with `OK`. Because the exact test patch is an entirely new file, this target run provides F2P coverage only and does not claim an independent P2P result. The dirty `/workspace/Paddle` checkout was not modified.
