# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #77064.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `1a6a9ab02e12fd792d036dc78b94f46a1371e6fa` (first parent of squash commit `407e3b6931a282a78653d559e675a598153ae977`)
- Resource: CPU; CUDA is optional and not required for acceptance.
- Platform: Linux x86_64 with a Python version supported by this revision, CMake, a compatible C++ compiler, NumPy, and pytest.
- Build path: **source build required**. The solution modifies C++ pybind argument preprocessing and API-generation metadata, so a prebuilt wheel or Python-only overlay cannot expose the fixed binding. Rebuild Paddle after applying the solution patch so code generation and C++ compilation run again.

## Verified Author Environment

The original change was built and tested on:

- OS: Windows 11 Home
- CPU/GPU: AMD 9800X3D + NVIDIA RTX 5070 Ti
- Python: 3.12
- CMake: 3.18.6
- Toolchain: Visual Studio 2022
- CUDA / cuDNN: 12.9 / 9.12.0

The target tests require CPU only. On CUDA-enabled builds, the upstream compatibility test also repeats its assertions when CUDA is available, but GPU execution is not an acceptance requirement.

## Run / Test / Fix Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build Paddle from source and make the built package importable.
3. Apply `tests/test.patch` from this task package (for example, set `TASK_DIR` to this directory).
4. From the Paddle source root, run `bash "$TASK_DIR/tests/test.sh"`. The new function aliases, Tensor-method alias, and compatibility entrypoint should fail or error before the fix; existing regression cases should remain passing.
5. Apply `solution/code.patch`.
6. Rebuild Paddle from source to regenerate and compile the public binding.
7. From the Paddle source root, run `bash "$TASK_DIR/tests/test.sh"` again. All target and regression tests should pass.

## Minimal F2P Command

```bash
python -m pytest \
  test/legacy_test/test_allclose_op.py::TestAllcloseAlias \
  test/legacy_test/test_compat_allclose.py \
  -q
```

`tests/test.sh` intentionally runs the complete existing allclose module as a P2P regression guard in addition to the new compatibility test file.

## Known Risks

- A rebuild is mandatory after the solution patch; otherwise the generated C++ binding remains stale and alias tests can still fail.
- There is no fixed Docker image verified against this historical revision, so maintainers must use an era-compatible source-build toolchain.
- The target tests use fixed small tensors and no external data or network access. CUDA branches are conditional and are not part of the pass requirement.
- The package records the merged fix and its test updates as-is. Verifiers should preserve the Run/Test/Fix order and must not test against an unrelated installed Paddle wheel.
