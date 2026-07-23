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

Build Paddle from source after applying `solution/code.patch`. A CPU build is sufficient for benchmark acceptance; GPU hardware is not required. Keep the build configuration capable of running legacy Python operator tests and PIR symbolic-shape tests. The upstream test module also contains a CINN dynamic-shape case, so enable CINN when the target environment supports that test dependency. If CINN is unavailable, report that limitation rather than treating a skipped or uncollectable case as complete verification.

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
```

At this state the new test module is present, but the API and compiled operator do not exist. Running the target tests should fail during import, API lookup, graph construction, or execution. The existing reduction P2P tests should remain usable with the base build.

Then apply the implementation and rebuild:

```bash
git apply "$TASK_DIR/solution/code.patch"
# Re-run CMake if needed, then rebuild and reinstall Paddle.
cmake --build build --parallel
python -m pip install --no-deps --force-reinstall build/python/dist/*.whl
bash "$TASK_DIR/tests/test.sh"
```

If the build system produces a wheel tagged for a different Python ABI, install and run tests with the matching interpreter. Do not rename the wheel to bypass ABI checks.

## Exact target tests

`tests/test.sh` runs:

```bash
python -m pytest test/legacy_test/test_aminmax_op.py -q
python -m pytest \
  test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest \
  -q
```

Expected post-fix results are passing forward, gradient, API compatibility, static/dynamic, output-tensor, dynamic-shape, and symbolic-shape cases. Stable pre-existing tests for `min`, `max`, `amin`, and `amax` can be added by the verifier as P2P regression guards.

## Limitations

The benchmark's required backend is CPU. The gold patch retains upstream GPU registrations, but GPU execution is not required for acceptance. No external dataset, network service, distributed topology, or multiple devices are needed.
