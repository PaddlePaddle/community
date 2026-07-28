# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `555b4a95615a35b301f348e081e56435a6d75da6`
- Gold commit: `01b7cdd95813a88bca9569f55328c4f6f0e675cb`
- Resource: CPU
- GPU required: no
- Suggested platform: Linux x86_64 with a Python version, CMake, C++ compiler, NumPy, and pytest supported by the base revision.
- Container or pinned external wheel: none provided.

## Mandatory Build Requirement

A source rebuild is required. The solution changes pybind C++ code and Python API YAML consumed by Paddle's code-generation and compilation pipeline. Apply the patches in the source checkout, rerun the normal code-generation/build process, and ensure the tested Python interpreter imports the resulting package.

A Python source overlay, an arbitrary installed wheel, or a stale build directory cannot verify this task. A locally built wheel is acceptable only when it was produced from the exact patched base revision.

## Patch And Test Order

Run from the root of a clean Paddle checkout at the exact base commit:

1. Apply `tests/test.patch`.
2. Build the base-with-tests state if checking F2P behavior.
3. Run the compatibility class. Positional and `x=` calls should remain valid, while `input=` should fail in both dynamic and static graph paths before the solution.
4. Apply `solution/code.patch`.
5. Regenerate and rebuild Paddle from source.
6. Run `bash tests/test.sh`. Both the F2P compatibility class and the complete P2P operator module should pass.

Example patch commands when the task package is available at `$TASK_DIR`:

```bash
git apply "$TASK_DIR/tests/test.patch"
git apply "$TASK_DIR/solution/code.patch"
```

## Target Commands

```bash
python -m pytest \
  test/legacy_test/test_api_compatibility.py::TestPixelShuffleAPI_Compatibility \
  -q

python -m pytest test/legacy_test/test_pixel_shuffle_op.py -q
```

The equivalent strict wrapper is:

```bash
bash tests/test.sh
```

## Verification Scope

- F2P: `TestPixelShuffleAPI_Compatibility`, covering positional, `x=`, and `input=` calls in dynamic and static graph modes.
- P2P: the complete `test/legacy_test/test_pixel_shuffle_op.py` module, covering numerical output, shape/layout, dtypes, gradients, and validation semantics.

## Known Risks

- Generated bindings can remain stale if code generation or compilation is skipped.
- Importing an unrelated installed Paddle wheel can produce misleading results; confirm `paddle.__file__` points to the intended build.
- Patch context and generated API metadata are tied to the exact base commit.
- CPU execution is sufficient, but Paddle's full source build still requires the normal native build toolchain.
- The upstream squash commit contains unrelated `paddle.unique` cleanup. It is deliberately absent from both task patches.
