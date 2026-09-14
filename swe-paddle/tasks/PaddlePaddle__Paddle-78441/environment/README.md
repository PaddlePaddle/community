# Environment Notes

## Expected environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `35b36cca24a780061268d20d6abe512e758837e6`
- Gold commit: `156159726b64d8f85747de864fb3ce41ea1f3f2f`
- Primary resource: Linux x86_64 CPU
- Dependencies: a Python version supported by the base revision, NumPy, pytest, CMake, Ninja or Make, and a compatible C/C++ toolchain
- Patch type: C++, operator YAML/code generation, infermeta, PIR symbolic shape, and Python API metadata
- Source build required: yes

## Build requirements

Start with a clean checkout at the exact base commit and initialize its submodules. A release or nightly wheel is insufficient: the implementation adds compiled kernels and changes build-time operator schemas, generated bindings, infermeta, and symbolic-shape registration.

Build Paddle from source after applying `solution/code.patch`. A CPU build is sufficient for benchmark acceptance; GPU hardware is not required. `-DWITH_CINN=ON` is required, not optional: `TestAminmaxDynamicShape::test_all_dynamic` requests the CINN backend and `paddle.base.libpaddle.pir.apply_cinn_pass` raises `Unimplemented("... please compile PaddlePaddle with CINN")` without it, and the symbolic-shape node needs the same build option for `check_infer_symbolic_if_need` to be more than a no-op. If CINN is unavailable, report that limitation rather than treating a skipped or uncollectable case as complete verification.

A typical CPU configuration is:

```bash
cmake -S . -B build-78441-base \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPY_VERSION=3.10 \
  -DWITH_GPU=OFF \
  -DWITH_DISTRIBUTE=OFF \
  -DWITH_CINN=ON \
  -DWITH_TESTING=OFF \
  -DWITH_CPP_TEST=OFF \
  -DWITH_SETUP_INSTALL=OFF \
  -DBUILD_WHL_PACKAGE=ON
cmake --build build-78441-base --parallel 8
```

Use options compatible with the actual host and base revision. Install the resulting wheel or otherwise ensure tests import the freshly rebuilt package, not a previously installed Paddle.

## Patch and verification order

Use sibling checkouts named `paddledebug` and `community`; the latter must contain the **updated local task files**. Start from their parent directory. All commands after `cd paddledebug` run in the Paddle repository root. On Linux names are case-sensitive: if the directory is named `PaddleDebug`, change only the initial `cd`.

Save unrelated working-tree changes before checkout. Use fresh `build-78441-base` and `build-78441-gold` directories to preserve the base wheel and avoid stale generated files/wheels. Do not delete an existing working tree or build to follow these instructions.

Use a dedicated Python 3.10 environment without another Paddle distribution installed, where `python` invokes that interpreter, with compatible GCC/G++, CMake 3.x (at least 3.18), Ninja, patchelf, Python development headers, and zlib development headers installed. The following source-derived configuration still needs to be confirmed on the actual Linux host.

### 1. Checkout and build base

```bash
cd paddledebug
git status --short
git checkout --detach 35b36cca24a780061268d20d6abe512e758837e6
git rev-parse HEAD
git submodule update --init --recursive
python --version
python -m pip install -r python/requirements.txt -r paddle/scripts/compile_requirements.txt
python -m pip install pytest wheel setuptools pyyaml ninja

cmake -S . -B build-78441-base -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPY_VERSION=3.10 \
  -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_CINN=ON \
  -DWITH_TESTING=OFF -DWITH_CPP_TEST=OFF \
  -DWITH_SETUP_INSTALL=OFF -DBUILD_WHL_PACKAGE=ON
cmake --build build-78441-base --parallel 8
python -m pip install --no-deps --force-reinstall build-78441-base/python/dist/*.whl
python -c "import paddle; print(paddle.__file__); print(paddle.version.commit); assert paddle.is_compiled_with_cinn(); assert not hasattr(paddle, 'aminmax')"
```

Verify CMake selected the same interpreter as `python`, the printed package path is the newly installed wheel, and the version commit is base. Each dedicated wheel directory should contain exactly one wheel compatible with that Python ABI. Eight parallel jobs are an example; adjust to available RAM.

### 2. Run: existing P2P guards on base

```bash
python -c "import sys; sys.path.insert(0, 'test'); sys.path.insert(0, 'test/legacy_test'); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))" \
  -q \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAPI_Compatibility::test_dygraph_Compatibility \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAminAPI_Compatibility::test_dygraph_Compatibility \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAminOutAPI::test_amax_out_in_dygraph \
  test/legacy_test/test_max_min_amax_amin_op.py::TestAmaxAminOutAPI::test_amin_out_in_dygraph
```

Expected: `4 passed`, exit code 0.

### 3. Test: add tests, retain the base build

```bash
git apply --check ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.patch
git apply ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.patch
git diff --check
mkdir -p ../aminmax-78441-logs
bash ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.sh > ../aminmax-78441-logs/test-1.log 2>&1
cat ../aminmax-78441-logs/test-1.log
bash ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.sh > ../aminmax-78441-logs/test-2.log 2>&1
cat ../aminmax-78441-logs/test-2.log
```

Both wrapper runs must exit **1**. Run these commands interactively, not in an outer `set -e` script that stops after the expected failure. The four P2P nodes must pass; 25 active legacy nodes and one symbolic node must fail because the API/op is absent. The empty float32 placeholder must be skipped. Dependency/import/collection failures do not demonstrate F2P behavior. OpTest teardown can add error reports: count unique test nodes, not the sum of failure/error headings.

### 4. Fix: apply implementation, build and install gold

Keep HEAD at base and retain the test patch; do not checkout gold over the corrected tests.

```bash
git apply --check ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/solution/code.patch
git apply ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/solution/code.patch
git diff --check
cmake -S . -B build-78441-gold -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPY_VERSION=3.10 \
  -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_CINN=ON \
  -DWITH_TESTING=OFF -DWITH_CPP_TEST=OFF \
  -DWITH_SETUP_INSTALL=OFF -DBUILD_WHL_PACKAGE=ON
cmake --build build-78441-gold --parallel 8
python -m pip install --no-deps --force-reinstall build-78441-gold/python/dist/*.whl
python -c "import paddle; print(paddle.__file__); print(paddle.version.commit); assert paddle.is_compiled_with_cinn(); assert hasattr(paddle, 'aminmax'); assert hasattr(paddle.Tensor, 'aminmax')"
bash ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.sh > ../aminmax-78441-logs/fix-1.log 2>&1
cat ../aminmax-78441-logs/fix-1.log
bash ../community/swe-paddle/tasks/PaddlePaddle__Paddle-78441/tests/test.sh > ../aminmax-78441-logs/fix-2.log 2>&1
cat ../aminmax-78441-logs/fix-2.log
```

Both Fix runs must exit **0**, with `4 passed`, `25 passed, 1 skipped`, and `1 passed`. This is **26 effective F2P / 4 P2P**, plus one excluded skip. Because the implementation is applied as a working-tree patch, the built version's commit can still report base; identify Fix using the patched sources, separate build/wheel, and API checks together.

No shell variable assignments or manual `PYTHONPATH`/FLAGS exports are needed in this procedure. The existing wrapper supplies the per-suite environment internally. Task-local `.gitattributes` keeps patches and the Bash script in LF format when checked out on Windows.

If the build system produces a wheel tagged for a different Python ABI, install and run tests with the matching interpreter. Do not rename the wheel to bypass ABI checks.

## Exact target tests

`tests/test.sh` sets `PYTHONPATH` per suite and runs, in order:

1. Four P2P nodes from `test/legacy_test/test_max_min_amax_amin_op.py`:
   - `TestAmaxAPI_Compatibility::test_dygraph_Compatibility`
   - `TestAminAPI_Compatibility::test_dygraph_Compatibility`
   - `TestAmaxAminOutAPI::test_amax_out_in_dygraph`
   - `TestAmaxAminOutAPI::test_amin_out_in_dygraph`
2. F2P legacy target: 25 active cases in `test/legacy_test/test_aminmax_op.py`; its 26th collected node, `TestAminmaxOpFloat32::test_check_grad`, is explicitly skipped and excluded from both F2P and P2P.
3. F2P symbolic-shape target: `test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest`.

For the legacy suites, `PYTHONPATH` includes `test/legacy_test` and `test` so `op_test`, `utils`, and `white_list` imports resolve. For the symbolic-shape suite, `PYTHONPATH` includes `test/ir/pir/cinn` and `test/ir/pir/cinn/symbolic`, with the CINN/PIR directory first because it and `test/legacy_test` both contain a module named `utils.py`.

The wrapper runs both F2P suites even when the first one fails, so the Base run records both F2P roles before exiting nonzero.

Expected post-fix results are passing forward, gradient, API compatibility, static/dynamic, output-tensor, dynamic-shape, and symbolic-shape cases, with all four amin/amax P2P nodes passing before and after the fix.

The upstream float32 gradient method contains only `pass`. Before the API exists, its inherited `setUp` fails; after the implementation is installed, the method returns without asserting anything. The test patch now adds an unconditional `unittest.skip` decorator, which skips the method before `setUp` in both states. The wrapper uses `-rs` to expose the reason. This decorator is the only intentional deviation from the upstream test diff. Float32 forward coverage and the existing substantive numerical and explicit gradient checks, including repeated extrema, remain unchanged.

## FLAGS required by the symbolic-shape node

`AminmaxOpInferSymbolicShapeTest` asserts on the `sym_shape_str` attribute that `check_infer_results` reads from each `pd_op.aminmax` operation. That attribute is written by `pir::shape::SetShapeAttrForOp` from the shape optimization pass, and on the `@to_static` path used by this test the pass is only added by `cinn::dialect::ir::CheckInferSymbolicIfNeed`, reached through `paddle.base.libpaddle.pir.check_infer_symbolic_if_need`. Two conditions gate it:

- the binding compiles to `// Do nothing.` unless `PADDLE_WITH_CINN` is defined, so the build must be configured with `-DWITH_CINN=ON`;
- `CheckInferSymbolicIfNeed` returns early unless `FLAGS_prim_forward`, `FLAGS_prim_backward` and `FLAGS_check_infer_symbolic` are all set. Setting the environment variable `FLAGS_prim_all=True` sets both prim flags.

When either condition is missing the node fails with `KeyError: 'sym_shape_str'` on the gold revision as well, which is a harness gap rather than a gap in `solution/code.patch`: that patch does implement `AminmaxOpInferSymbolicShape` and does add `paddle::dialect::InferSymbolicShapeInterface` to the op in `ops.yaml`.

`tests/test.sh` therefore supplies the same FLAGS environment that `test/ir/pir/cinn/symbolic/CMakeLists.txt` uses for this file: `FLAGS_check_infer_symbolic=1`, `FLAGS_enable_pir_api=1`, `FLAGS_prim_enable_dynamic=true`, `FLAGS_prim_all=True`, and `FLAGS_cinn_new_group_scheduler=1`. No manual exports are needed.

Upstream additionally wraps that whole `CMakeLists.txt` in `if(WITH_GPU)` and labels the tests `RUN_TYPE=CINN`, so a GPU + CINN build is the configuration the node was written against. A CPU build with `-DWITH_CINN=ON` is the minimum this task needs.

## Static verification

- Both patches apply in order to exact base files, with LF and Windows CRLF checkouts.
- `solution/code.patch` remains the exact upstream non-test diff, and the P2P source file is byte-for-byte identical at base and gold.
- The explicit skip decorator is the only behavioral change to the upstream tests; the new-file hunk length and blob hash were updated accordingly.
- Python syntax, inherited test counts (26 legacy nodes, one excluded skip), skip-before-setup behavior, Bash syntax, and `git diff --check` passed.

The review that identified the empty placeholder reported two successful full gold runs of the previous wrapper. The corrected wrapper retains both CINN-dependent nodes and the symbolic flags. The Windows results below cover this revision's available subset; full two-round Linux Run/Test/Fix verification of the revised package is still pending.

## Windows source-build verification supplied by the contributor

The contributor subsequently supplied one before/after round using the revised tests from the `PaddleDebug` checkout and Python 3.12. The command selected the four P2P nodes separately and ran the legacy suite with `-k "not TestAminmaxDynamicShape"`.

| Target | Before implementation patch | After implementation patch |
| --- | --- | --- |
| Existing amin/amax P2P nodes | `4 passed` | `4 passed` |
| Selected aminmax legacy nodes | `24 failed, 1 skipped, 1 deselected, 7 errors` | `24 passed, 1 skipped, 1 deselected, 6 warnings` |

All 24 unique failed nodes reported missing `paddle.aminmax`. The seven additional errors came from `OpTest.tearDownClass` after setup/execution had aborted before establishing class-level operator metadata; they belong to seven of those same failed nodes, not seven additional F2P cases. They disappeared after installing the implementation build. The six post-fix warnings concern the PIR Tensor `place` interface and did not fail any tests.

The placeholder was explicitly skipped in both runs. `TestAminmaxDynamicShape::test_all_dynamic` was deselected, and the separate `AminmaxOpInferSymbolicShapeTest` was not invoked. These logs establish **24 effective Windows F2P / 4 P2P for one round**, not full 26/4 acceptance or two-round verification.

Read-only checks after receiving the logs confirmed HEAD at `35b36cca24a780061268d20d6abe512e758837e6`, both task patches applied (reverse application checks passed), CMake `WITH_GPU=ON` / `WITH_CINN=OFF`, and the installed runtime reporting `3.5.0.dev20260427`, the base commit, `aminmax=True`, `CUDA=True`, and `CINN=False`. The version commit remains base because the implementation was applied as a working-tree patch.

## Remaining acceptance scope

The benchmark's required backend is CPU. The gold patch retains upstream GPU registrations, but GPU execution is not required for acceptance. No external dataset, network service, distributed topology, or multiple devices are needed.
