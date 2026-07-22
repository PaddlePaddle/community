# Environment Notes

## Expected environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `c3a5e799eb9e390f830c9e6c3fbea2e9370afa7f`
- Gold commit: `ddb483237539d4d23c7dbbd44e3a360439c780ed`
- Patch type: Python production and test files
- Python dependency: `pytest`
- Complete resource requirement: one CUDA or XPU device with a matching Paddle source build

## Checkout and build

Start from a clean checkout of the exact base commit. The tests import Paddle's compiled runtime and instantiate device places, so use a Paddle installation built from this checkout (or an equivalent build whose Python source is overlaid by this checkout). A source build is recommended to keep the Python API and compiled extension aligned with the historical revision.

Follow Paddle's build prerequisites for the selected backend. CUDA validation requires a CUDA-enabled build and compatible CUDA/cuDNN runtime; XPU validation requires an XPU-enabled build and runtime. No dataset, network service, multi-device topology, or distributed setup is required.

## Device limitations

- CUDA builds can validate CUDA-place conversion, CUDA pinned allocation, CPU-to-CUDA-pinned behavior, and CUDA-conditional API tests.
- XPU builds can validate XPU-place conversion, XPU pinned allocation, and CPU-to-XPU-pinned behavior.
- CPU-only builds have no pinned allocator. They can validate the explicit `Pinning memory is not supported` path and some mocked branch coverage, but accelerator-dependent tests are skipped or cannot allocate pinned memory. A green CPU-only run is therefore not complete task verification.

## Patch application order

From the Paddle repository root, assuming this task directory is available as `$TASK_DIR`:

```bash
git checkout c3a5e799eb9e390f830c9e6c3fbea2e9370afa7f
git apply "$TASK_DIR/tests/test.patch"
bash "$TASK_DIR/tests/test.sh"       # expected to expose failures before the fix
git apply "$TASK_DIR/solution/code.patch"
bash "$TASK_DIR/tests/test.sh"       # expected to pass on a compatible build/device
```

Apply the test patch before the solution patch. Both patches are rooted at the Paddle repository root.

## Exact test command

`tests/test.sh` runs:

```bash
python -m pytest \
  test/legacy_test/test_to_pinned_place.py \
  test/legacy_test/test_eager_tensor.py \
  test/legacy_test/test_rand.py \
  test/legacy_test/test_randint_op.py \
  test/legacy_test/test_randperm_op.py \
  -q
```

Run it from the root of the built Paddle checkout. Before applying the solution, new tests requiring the missing behavior should fail on a compatible accelerator build. After applying the solution, all applicable tests should pass; backend-conditional skips are expected when that backend is unavailable.
