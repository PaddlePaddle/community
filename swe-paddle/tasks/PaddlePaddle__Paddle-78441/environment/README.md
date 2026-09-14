# Environment Notes

## Expected environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `35b36cca24a780061268d20d6abe512e758837e6`
- Gold commit: `156159726b64d8f85747de864fb3ce41ea1f3f2f`
- Acceptance backend: Linux x86_64 CPU with `WITH_CINN=ON`
- Dependencies: a Python version supported by base, NumPy, pytest, and a compatible Paddle source-build toolchain

## Build requirements

Build and install Paddle from the exact base sources, then rebuild and reinstall after applying the implementation patch. A release or nightly wheel cannot establish this task's before/after behavior: the change adds compiled kernels and updates operator schemas, generated bindings, infermeta, and symbolic-shape registration. Use the same build options and Python interpreter in both stages.

A CPU build is sufficient. `WITH_CINN=ON` is required because `TestAminmaxDynamicShape::test_all_dynamic` requests the CINN backend and the separate symbolic-shape test needs the CINN shape-optimization path. The historical CINN dependencies use Linux archives; native Windows can validate the remaining subset but cannot complete acceptance.

## Base → Test → Fix

Run from the Paddle checkout root. `TASK_DIR` denotes the absolute path to this task package, supplied by the runner; the task package and checkout may have any directory names or relative locations.

### Base

```bash
git checkout 35b36cca24a780061268d20d6abe512e758837e6
git submodule update --init --recursive
```

Build and install this revision, then run the four existing amin/amax P2P nodes listed in the task README. They must pass before applying the patches.

### Test

```bash
git apply --check "$TASK_DIR/tests/test.patch"
git apply "$TASK_DIR/tests/test.patch"
bash "$TASK_DIR/tests/test.sh"
```

Keep using the base build. The four P2P nodes must pass; the 25 legacy F2P nodes and one symbolic F2P node must fail because the API/op is absent. The wrapper runs both F2P suites and exits nonzero. Dependency or collection failures do not establish F2P behavior.

### Fix

```bash
git apply --check "$TASK_DIR/solution/code.patch"
git apply "$TASK_DIR/solution/code.patch"
```

Rebuild and reinstall Paddle from the patched sources, retaining the test patch, then run:

```bash
bash "$TASK_DIR/tests/test.sh"
```

The expected summaries are `4 passed`, `25 passed`, and `1 passed`, with wrapper exit code 0: **26 F2P / 4 P2P, 30 collected nodes**. Repeat Test/Fix runs when collecting two-round acceptance evidence.

## Test collection and runtime settings

The empty `TestAminmaxOpFloat32::test_check_grad` method is removed. The class sets `test_check_grad = None` to mask the inherited method; simply deleting the override would collect the parent's numerical gradient test instead. The placeholder produces neither a collected node nor a skip. The float32 forward test and all other gradient assertions remain unchanged.

The wrapper supplies separate import paths for the legacy and symbolic suites because both contain a `utils.py`. It also retains the upstream symbolic flags: `FLAGS_check_infer_symbolic=1`, `FLAGS_enable_pir_api=1`, `FLAGS_prim_enable_dynamic=true`, `FLAGS_prim_all=True`, and `FLAGS_cinn_new_group_scheduler=1`. Together with CINN, these enable the pass that attaches the asserted `sym_shape_str` attribute; omitting them can cause `KeyError: 'sym_shape_str'` on gold.

## Verification scope

Static checks verify exact-base patch application, unchanged upstream gold contents, Python/Bash syntax, and inherited test collection. The P2P source file is identical at base and gold.

The contributor's Windows source-build logs established 24 legacy F2P transitions and four P2P passes with Python 3.12, `WITH_GPU=ON`, and `WITH_CINN=OFF`. Those earlier logs used the skip version of the placeholder; base's seven extra teardown errors belonged to the same failed nodes and were not additional F2P cases. With the placeholder removed, the same gold runtime was checked again: `24 passed, 1 deselected` for the legacy subset and `4 passed` for P2P, with no skips. Collection checks found 25 legacy nodes and one separate symbolic node. The revised patch has not been rerun on a base runtime; full Linux/CINN execution remains outstanding for the two CINN-dependent nodes.
