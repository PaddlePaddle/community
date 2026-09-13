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
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPY_VERSION=3.10 \
  -DWITH_GPU=OFF \
  -DWITH_DISTRIBUTE=OFF \
  -DWITH_CINN=ON \
  -DWITH_TESTING=OFF
ninja -j"$(nproc)"
```

Use options compatible with the actual host and base revision. Install the resulting wheel or otherwise ensure tests import the freshly rebuilt package, not a previously installed Paddle.

## Patch and verification order

From the Paddle repository root, with this task directory available as `$TASK_DIR`:

```bash
git checkout 35b36cca24a780061268d20d6abe512e758837e6
git submodule update --init --recursive
git apply "$TASK_DIR/tests/test.patch"
PYTHON_BIN=python bash "$TASK_DIR/tests/test.sh"
```

At this state the new test module is present, but the API and compiled operator do not exist. The wrapper must first pass the four existing amin/amax P2P nodes, then fail while the F2P targets import, create, or execute the missing operation.

Then apply the implementation and rebuild:

```bash
git apply "$TASK_DIR/solution/code.patch"
# Re-run CMake if needed, then rebuild and reinstall Paddle.
cmake --build build --parallel
python -m pip install --no-deps --force-reinstall build/python/dist/*.whl
PYTHON_BIN=python bash "$TASK_DIR/tests/test.sh"
```

After the rebuild, all P2P nodes and all F2P targets must pass.

If the build system produces a wheel tagged for a different Python ABI, install and run tests with the matching interpreter. Do not rename the wheel to bypass ABI checks.

## Exact target tests

`tests/test.sh` sets `PYTHONPATH` per suite and runs, in order:

1. Four P2P nodes from `test/legacy_test/test_max_min_amax_amin_op.py`:
   - `TestAmaxAPI_Compatibility::test_dygraph_Compatibility`
   - `TestAminAPI_Compatibility::test_dygraph_Compatibility`
   - `TestAmaxAminOutAPI::test_amax_out_in_dygraph`
   - `TestAmaxAminOutAPI::test_amin_out_in_dygraph`
2. F2P legacy target: the 26 cases in `test/legacy_test/test_aminmax_op.py`.
3. F2P symbolic-shape target: `test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest`.

For the legacy suites, `PYTHONPATH` includes `test/legacy_test` and `test` so `op_test`, `utils`, and `white_list` imports resolve. For the symbolic-shape suite, `PYTHONPATH` includes `test/ir/pir/cinn` and `test/ir/pir/cinn/symbolic`, with the CINN/PIR directory first because it and `test/legacy_test` both contain a module named `utils.py`.

The wrapper runs both F2P suites even when the first one fails, so the Base run records both F2P roles before exiting nonzero.

Expected post-fix results are passing forward, gradient, API compatibility, static/dynamic, output-tensor, dynamic-shape, and symbolic-shape cases, with all four amin/amax P2P nodes passing before and after the fix.

## FLAGS required by the symbolic-shape node

`AminmaxOpInferSymbolicShapeTest` asserts on the `sym_shape_str` attribute that `check_infer_results` reads from each `pd_op.aminmax` operation. That attribute is written by `pir::shape::SetShapeAttrForOp` from the shape optimization pass, and on the `@to_static` path used by this test the pass is only added by `cinn::dialect::ir::CheckInferSymbolicIfNeed`, reached through `paddle.base.libpaddle.pir.check_infer_symbolic_if_need`. Two conditions gate it:

- the binding compiles to `// Do nothing.` unless `PADDLE_WITH_CINN` is defined, so the build must be configured with `-DWITH_CINN=ON`;
- `CheckInferSymbolicIfNeed` returns early unless `FLAGS_prim_forward`, `FLAGS_prim_backward` and `FLAGS_check_infer_symbolic` are all set. Setting the environment variable `FLAGS_prim_all=True` sets both prim flags.

When either condition is missing the node fails with `KeyError: 'sym_shape_str'` on the gold revision as well, which is a harness gap rather than a gap in `solution/code.patch`: that patch does implement `AminmaxOpInferSymbolicShape` and does add `paddle::dialect::InferSymbolicShapeInterface` to the op in `ops.yaml`.

`tests/test.sh` therefore runs the node with the same FLAGS environment that `test/ir/pir/cinn/symbolic/CMakeLists.txt` uses for this file:

```bash
FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 \
  FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True \
  FLAGS_cinn_new_group_scheduler=1
```

Upstream additionally wraps that whole `CMakeLists.txt` in `if(WITH_GPU)` and labels the tests `RUN_TYPE=CINN`, so a GPU + CINN build is the configuration the node was written against. A CPU build with `-DWITH_CINN=ON` is the minimum this task needs.

## Local validation note

- `test/legacy_test/test_max_min_amax_amin_op.py` is byte-for-byte unchanged between the base and gold revisions.
- With an installed compatible CPU runtime and the exact base test sources, the four selected P2P nodes passed (`4 passed`).
- With the legacy `PYTHONPATH` configured by `tests/test.sh`, `test_aminmax_op.py` collected successfully and then failed at `AttributeError: module 'paddle' has no attribute 'aminmax'`, which is the expected base-like signal on a runtime without the gold patch.
- With the CINN/PIR `PYTHONPATH` configured by `tests/test.sh`, `AminmaxOpInferSymbolicShapeTest` collected successfully. Running it without the FLAGS environment reproduces `KeyError: 'sym_shape_str'` regardless of which revision is installed, which is why `tests/test.sh` now exports the same flags that upstream's CTest registration uses.

The FLAGS change itself was derived from the gold revision's source rather than from a local run. CINN cannot be built on Windows, where the work was done: `cmake/cinn/external/llvm.cmake`, `isl.cmake` and `ginac.cmake` fetch glibc-only Linux archives, and the available local build reports `paddle.is_compiled_with_cinn() == False`, so the node cannot reach a passing state there under any flag combination. What the source does establish is that the gate in `CheckInferSymbolicIfNeed` is the only thing standing between the node and the attribute it reads; that `aminmax` has no prim decomposition rule, so `FLAGS_prim_all` cannot remove `pd_op.aminmax` from the program and make the assertion vacuous; and that the base revision exposes no `paddle.aminmax` at all, so the node stays Base-red with or without the flags. The preceding validation round also reported `TestAminmaxDynamicShape::test_all_dynamic` as Gold-green, and that case requests `backend="CINN"` while `ApplyCinnPass` raises `Unimplemented` without `PADDLE_WITH_CINN`, which indicates the validation environment does have CINN compiled in. A confirming two-round run on such an environment is still required.

If the node still fails there, the fallback is to drop it from the F2P set and reclassify it rather than to make it conditional: a node that disappears when CINN is absent would make the F2P set depend on the environment. The symbolic-shape registration it targets would stay covered by `test_aminmax_op.py::TestAminmaxInferSymbolicShapePass::test_infer_symbolic_shape_pass`, which calls `paddle.base.libpaddle.pir.infer_symbolic_shape_pass` directly and needs neither a CINN build nor extra flags.

## Limitations

The benchmark's required backend is CPU. The gold patch retains upstream GPU registrations, but GPU execution is not required for acceptance. No external dataset, network service, distributed topology, or multiple devices are needed.
