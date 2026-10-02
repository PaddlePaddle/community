# Environment Notes

## Expected environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ad1d2d4df4731d62fe41263e276bb7d7f30e16e7`
- Gold commit: `718431cf276c9bd32c089ee2daf6cc7d54af2aa8` (squash merge of PR #79391)
- Primary resource: Linux x86_64 CPU
- GPU required: no
- Dependencies: a Python version supported by the base revision (the PR was developed on 3.10), NumPy, pytest
- Patch type: pure Python (`python/paddle/compat/**` only)
- Source rebuild required: **no**

## Preconditions

**PyTorch must not be importable.** `test/compat/test_torch_proxy.py` asserts that `import torch` raises
`ModuleNotFoundError` whenever compat is off, so it only holds in a torch-free interpreter — which is what
upstream Paddle CI provides, and what `test/compat/test_torch_proxy_mixed.py` then simulates for itself by
appending `test/compat/fake_torch_modules` to `sys.path`. If the verifier image ships a real PyTorch, that
P2P module fails independently of this task; uninstall torch or drop that node from the P2P set rather than
reading it as a regression.

## Build path

The gold patch touches only Python files under `python/paddle/compat/`. It adds no C++/CUDA/kernel/infermeta
code and no build-system entry — `test/compat/CMakeLists.txt` globs `test_*.py`, so the two new test modules
register themselves. A native rebuild is therefore not required.

What *is* required is that `import paddle` resolves to the **patched** Python sources. Either of the
following satisfies that:

- a source build of the base commit whose `build/python` (or installed package) is refreshed after each
  `git apply`, or
- an installed CPU wheel built from a commit compatible with the base revision, with the repository's
  `python/paddle` tree overlaid onto the installed package (see "Wheel overlay" below).

Confirm which files are actually loaded before trusting a result:

```bash
python -c "import paddle, paddle.compat.proxy as p; print(paddle.__file__); print(p.__file__)"
```

Both paths must sit inside the tree the patches were applied to. A wheel whose `paddle/compat/` predates the
base revision produces failures that have nothing to do with this task.

### Wheel overlay

Because the patch is pure Python, a wheel close to the base commit can supply the compiled runtime while the
checkout supplies every `.py`:

```bash
python -c "import zipfile; zipfile.ZipFile('<wheel>').extractall('overlay')"
git -C <paddle-repo> archive <commit> python/paddle | tar -x -C repo
cp -r repo/python/paddle/. overlay/paddle/          # copy over, never delete
PYTHONPATH=overlay python -m pytest -q test/compat/...
```

Copy over the extracted wheel rather than replacing it. 15 generated modules exist only in the wheel
(`version/__init__.py`, `base/proto/*_pb2.py`, `base/dygraph/generated_tensor_methods_patch.py`,
`incubate/autograd/phi_ops_map.py`, `cuda_env.py`, ...) and must survive, while ~26 source modules
(`nn/modules/utils.py`, `optim/*`, ...) exist only in the checkout and must be added — `paddle/compat/nn`
imports `paddle.nn.modules.utils._single`, so overlaying just `compat/` is not sufficient.

## Patch and verification order

From the Paddle repository root, with this task directory available as `$TASK_DIR`:

```bash
git checkout ad1d2d4df4731d62fe41263e276bb7d7f30e16e7
git apply "$TASK_DIR/tests/test.patch"
PYTHON_BIN=python bash "$TASK_DIR/tests/test.sh"
```

At this state both P2P torch-proxy modules pass, and both F2P modules fail:

- `test_compat_namespace_aliased.py` fails at collection with
  `ModuleNotFoundError: No module named 'paddle.compat.api_dispatch'`.
- `test_compat_level2_internal_composites.py` fails in `setUpModule` with
  `TypeError: enable_compat() got an unexpected keyword argument 'level'`, erroring all 10 cases.

The wrapper exits `1`.

Then apply the implementation and re-run:

```bash
git apply "$TASK_DIR/solution/code.patch"
# Refresh the importable paddle package if it is not the source tree itself.
PYTHON_BIN=python bash "$TASK_DIR/tests/test.sh"
```

All four modules must now pass and the wrapper must exit `0`.

`base + tests/test.patch + solution/code.patch` reproduces the gold tree
`65f0475d23a8a7622ccaa6414136603be8ab3ba7` exactly, so the two patches together are the merged PR with
nothing added and nothing left out.

## Exact target tests

`tests/test.sh` runs four modules, each in its own interpreter:

| Role | Module |
| --- | --- |
| P2P | `test/compat/test_torch_proxy.py` |
| P2P | `test/compat/test_torch_proxy_mixed.py` |
| F2P | `test/compat/test_compat_namespace_aliased.py` |
| F2P | `test/compat/test_compat_level2_internal_composites.py` |

Process isolation is mandatory, not cosmetic. Each module mutates process-global state:

- `test_torch_proxy_mixed.py` puts `test/compat/fake_torch_modules` on `sys.path`, which makes a *real*
  (fake) `torch` package importable. `test_torch_proxy.py` asserts that `import torch` raises
  `ModuleNotFoundError` while compat is off, so the two modules cannot share an interpreter.
- `test_compat_level2_internal_composites.py` enables `level=2` for the whole module in `setUpModule` and
  only restores it in `tearDownModule`.
- `test_compat_namespace_aliased.py` rewrites `paddle.*`, `paddle.Tensor` methods, `sys.meta_path` and
  `sys.modules["torch*"]`, and its `setUp` drains the proxy finder before every case.

This mirrors upstream, where `test/compat/CMakeLists.txt` registers one `py_test_modules` target per file.

No `PYTHONPATH` is needed: both new modules append `test/compat/fake_modules` to `sys.path` themselves, and
neither imports `op_test` or the `white_list` helpers.

Equivalent direct invocation, if pytest is unavailable:

```bash
python test/compat/test_torch_proxy.py
python test/compat/test_torch_proxy_mixed.py
python test/compat/test_compat_namespace_aliased.py
python test/compat/test_compat_level2_internal_composites.py
```

## Local validation note

Validated by loading the base and then base+gold Python sources over a compiled Paddle runtime from a commit
eight days before the base revision, on CPU, with a torch-free `sys.path`:

| Module | Role | Base | Base + gold patch |
| --- | --- | --- | --- |
| `test_torch_proxy.py` | P2P | 12 passed | 12 passed |
| `test_torch_proxy_mixed.py` | P2P | 4 passed | 4 passed |
| `test_compat_namespace_aliased.py` | F2P | collection error (`paddle.compat.api_dispatch` missing) | 29 passed, 14 subtests passed |
| `test_compat_level2_internal_composites.py` | F2P | 10 errors (`enable_compat()` has no `level`) | 10 passed |

Wrapper exit codes were checked separately with stub interpreters: `0` when all four modules pass, `1` when
the F2P modules fail, and both F2P modules always execute so a Base log records the full role matrix.

## Notes and risks

- `test_compat_namespace_aliased.py` pins the device to CPU in `setUp` and restores it in `tearDown`; the
  upstream comment records a DCU GPU-op hang in the attention/SDPA path. CPU-only execution is both
  sufficient and the intended configuration.
- Two cases use `unittest.subTest` (14 sub-cases total). With `pytest-subtests` installed they are reported
  individually; without it pytest does not advertise `addSubTest`, so a failing sub-case propagates and
  fails the parent test — also a correct outcome. The plugin only changes reporting granularity.
- `level=2` mutates process-wide state. Every case in the target modules pairs `enable_compat` with
  `disable_compat`, but a harness that shuffles or parallelises cases *within* a module can still leak
  state: do not run these modules under `pytest-randomly` or `pytest-xdist` in a shared interpreter.
- The gold patch changes `_register_compat_override` so the `paddle.compat` root package is walked too. That
  adds `torch.sort`/`min`/`max`/`split`/`unique`/`median`/`nanmedian`/`allclose`/`equal`/`seed`/`slogdet`
  overrides to the torch proxy at **both** levels. The two P2P modules assert only on
  `torch.sin`/`cos`/`relu`/`nn.Conv2d`/`nn.functional.sigmoid`/`nn.Unfold`/`nn.Linear`/`TorchVersion`, none
  of which is a root compat symbol, so they are unaffected — confirmed by the runs above.
- `test/compat/fake_modules/torch_proxy_root_api_module.py` ships with the test patch for fidelity with the
  merged PR, but nothing in the gold tree references it and nothing auto-imports it (`CMakeLists.txt` globs
  `test_*.py` only). It is inert.
- On Windows checkouts with `core.autocrlf=true`, both patches apply as shipped; `git apply` normalises line
  endings on both sides. `git apply --3way` works as a fallback because every blob named by the `index`
  lines is present in Paddle history.
- No external dataset, network access, distributed topology, or multiple devices are required.
