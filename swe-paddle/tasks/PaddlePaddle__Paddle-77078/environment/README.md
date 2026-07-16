# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #77078.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `f2de7486a07cbdbb6586771b5943df4bccc6d35c` (parent of squash-merge commit `78499bd`)
- Resource: CPU (the target test runs on CPU place; no GPU required for the target behavior)
- GPU required: no
- Build path: **source build required**. This patch modifies the eager auto code
  generator (`paddle/fluid/eager/auto_code_generator/generator/monkey_patch_gen.py`)
  and a build-time config (`paddle/phi/ops/yaml/python_api_info.yaml`). Both are
  consumed at compile time to generate the eager C++/pybind bindings, so a pure
  Python overlay / prebuilt wheel is **not** sufficient — the project must be
  rebuilt so code generation runs again.

## Verified Build Environment (author)

The original PR was authored, built, and tested on Windows:

- OS: Windows 11 Home
- CPU/GPU: AMD 9800X3D + NVIDIA RTX 5070Ti
- CUDA / cuDNN: 12.9 / 9.12.0
- Python: 3.12
- CMake: 3.18.6
- Toolchain: Visual Studio 2022

This verified environment includes a GPU, but the target test only exercises CPU.
The minimal Linux CPU-only build configuration and compile time have not been
measured yet — maintainers should build an era-matched environment and record the
confirmed minimum ("verified environment + unconfirmed minimum").

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit and build from source.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the new compatibility cases (`input=` alias, `out=`
   parameter, `paddle.linalg.inv` path) should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild** (code generation must re-run).
5. Run `bash tests/test.sh` again; all target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- Build-time codegen: the fix only takes effect after a rebuild; a Python-only
  reapply will not regenerate the eager bindings.
- `python/paddle/_paddle_docs.py` is prone to merge conflicts (flagged in PR
  review); watch the apply landing point.
- The complex-gradient cases and random-matrix invertibility depend on a fixed
  seed (`np.random.seed(123)` plus an invertibility check). The verifier should
  derive stable F2P / P2P node IDs from repeated runs.
