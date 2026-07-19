# Environment Notes

## Pinned Source

- Repository: `PaddlePaddle/Paddle`
- Base commit: `7743e779aff3e35b8bd748b2c69b9332f5d8dfd7`
- Reference squash commit: `3e4695db8fc3a19e9e055709941ad2b99f7f6c5f`
- Target backend: CPU
- GPU required: no

The original PR author built and tested the change on Windows 11 Home with Python 3.12, CMake 3.18.6, Visual Studio 2022, CUDA 12.9, and cuDNN 9.12.0. The author machine used a Ryzen 7 9800X3D and RTX 5070 Ti; that GPU is not a verifier requirement.

## Build Requirement

A clean source build is mandatory. The solution changes build-time API configuration, generated Python/C++ bindings, infermeta behavior, and compiled forward/backward kernels. Reusing an unrelated wheel, copying Python files over an installed package, or running against a build from another revision does not represent either the pinned base or fixed state.

The verifier should use its normal Paddle source-build procedure and ensure that the selected `PYTHON_BIN` imports the freshly built package. Although verification is CPU-only, the CUDA implementation section remains in the gold patch because it is part of the coherent merged solution.

## Run/Test/Fix Order

### Run the base

1. Check out the exact base commit in a clean Paddle worktree.
2. Apply `tests/test.patch` and build that source revision.
3. Run `bash tests/test.sh` from the Paddle repository root.
4. Classify existing positional/keyword, broadcasting, empty-Tensor, integer-output, and gradient nodes as pass-to-pass guards. Alias, `out`, and Tensor-method nodes should fail only with unsupported public-call errors such as `TypeError` or `AttributeError`; import, linkage, or environment failures are invalid results.

### Test the gold solution

1. Return to a clean checkout of the same base commit.
2. Apply `tests/test.patch`, then `solution/code.patch`.
3. Reconfigure or rebuild as required so build-time generation and all affected C++ sources are refreshed.
4. Run `bash tests/test.sh`; every selected node must pass.
5. Repeat the focused suite when practical to rule out global-mode contamination.

### Fix an attempted solution

1. Start from the pinned base with `tests/test.patch` applied.
2. Implement only the observable requirements in `instruction.md`.
3. Rebuild after changes that affect generated bindings or compiled operator behavior.
4. Run `bash tests/test.sh` and require all nodes to pass.

## Test Runner Notes

- `PYTHON_BIN` defaults to `python` and may be overridden.
- `test/legacy_test` is prepended to `PYTHONPATH` so legacy test utilities resolve.
- Every pytest node runs in a separate process because these tests switch Paddle between static and dynamic modes globally.
- The selected nodes use CPU behavior; no CUDA test is required.

## Verification Status and Risks

During package authoring, commit ancestry, patch boundaries, patch fidelity, Python syntax, shell syntax, and clean application against exact base file contents were checked. The nine fixed source files are also compared with their squash-commit versions during structural validation.

A Linux Paddle source configure/build and behavioral base/fixed pytest run were not executed in the community repository workspace because it does not contain a trustworthy Paddle source build for the pinned revision. Those steps remain pending for the SWE-Paddle Run/Test/Fix verifier and must not be inferred from structural checks.

Primary environment risks are stale generated bindings, importing a wheel instead of the fresh build, and legacy tests leaking global static/dynamic mode. The required rebuild, import-path check, and per-node pytest processes address these risks.
