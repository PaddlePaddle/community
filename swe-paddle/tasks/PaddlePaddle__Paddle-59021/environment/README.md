# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #59021.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `a53f40972d9dea85b44e6eae288f14c1bd01e3a7` (parent of #59021 merge `3af9eb7e`)
- Gold endpoint: `3af9eb7eb21f80e81f3573c427feeebbd621a72a`
- Resource: CPU
- GPU required: no
- Patch type: **source build required** (C++ adaptor / utils + YAML).
- Paddle install: source checkout at `base_commit`, build/install so PIR executor paths are available.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the SelectedRows `len` F2P cases should **fail / error** before the fix, while the P2P cases pass.
4. Apply `solution/code.patch` and **rebuild**.
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

`tests/test.sh` sets `PYTHONPATH=test/legacy_test:test` and runs three blocks. Every block runs even if an earlier one fails, and the script exits non-zero if any block fails.

| Block | Selection | Run settings | Role |
| --- | --- | --- | --- |
| 1 | `test/legacy_test/test_fuse_elewise_add_act_pass.py` | `FLAGS_enable_pir_in_executor=true`, `paddle.enable_static()` before pytest | P2P |
| 2 | `test_len.py::TestLen`, `test_len.py::TestLenWithTensorArray` | `FLAGS_enable_pir_with_pt_in_dy2st=0` | P2P |
| 3 | `test_len.py::TestLenWithSelectedRows` | `FLAGS_enable_pir_with_pt_in_dy2st=0`, `FLAGS_enable_pir_in_executor=true` | F2P |

Expected results:

- F2P (3): `TestLenWithSelectedRows::test_len__{ast,sot,sot_mgs10}_legacy_ir` fail before the fix and pass after it.
- P2P (9): `TestMNIST::test_simple_fc_with_fuse_op`, `TestMNIST::test_batchnorm_fc_with_fuse_op`, `TestFuseActElewiseAddInplaceGradPass::test_fuse_act_add_grad_pass_cpu`, and the three generated cases of `TestLen` and of `TestLenWithTensorArray`, pass before and after the fix.
- `TestFuseActElewiseAddInplaceGradPass::test_fuse_act_add_grad_pass_cuda` returns early without CUDA and is not counted as a pass.

## Run Settings

- `test_fuse_elewise_add_act_pass` is in `tools/static_mode_white_list.py`. Upstream CTest registers it as `test_fuse_elewise_add_act_pass_pir` with `FLAGS_enable_pir_in_executor=true` and runs it through `tools/test_runner.py`, which calls `paddle.enable_static()` before loading the module. pytest does not go through that runner, so `tests/test.sh` enables static mode explicitly.
- `FLAGS_enable_pir_with_pt_in_dy2st` defaults to `true`. Upstream lists `test_len` in `DISABLE_PIR_PT_MODES` in `test/dygraph_to_static/CMakeLists.txt` at both the base and gold revisions, so CTest runs it with `FLAGS_enable_pir_with_pt_in_dy2st=0`. A bare pytest run leaves the flag on, and `TestLenWithTensorArray` then crashes (SIGSEGV) in the PIR + PT dy2st path. That path is outside the scope of this task; the gold patch does not touch it.
- `TestLenWithSelectedRows` builds and runs a legacy program with `Executor` directly. The target behavior is only reached when the executor translates that program to PIR, which requires `FLAGS_enable_pir_in_executor=true`. Without it the case passes on the base revision as well.

## Known Risks

- `test_len` 本身未改文件；F2P 依赖 PIR executor flag，且必须同时关闭 `FLAGS_enable_pir_with_pt_in_dy2st`。
- `test_fuse_elewise_add_act_pass` 在 Apple 环境下被 CMake 门禁排除；CUDA 用例在无 GPU 时直接返回，不计为通过。
- 不要改回裸 `pytest` 入口：会丢失静态图初始化，并让 `test_len` 进入上游禁用的 PIR + PT 动转静模式。
