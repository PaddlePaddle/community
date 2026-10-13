# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #57827.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `8b1a29ba9bafc16116f97422574e85d208540332` (parent of #57827 merge `3ac5e693`)
- Gold endpoint: `3ac5e693b34eb3164fe076d489dc01bea9170843`
- Resource: CPU
- GPU required: no
- Patch type: **source build required** (C++ / YAML / infermeta / adaptor).
- Paddle install: source checkout at `base_commit`, build/install so PIR dy2static paths are available.
- Build option: `WITH_MKLDNN=ON`. `test_in_static_mode_mkldnn` only trains when `paddle.base.core.is_compiled_with_mkldnn()` is true; without oneDNN its body is empty and the node must not be counted as a pass.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the PIR F2P cases should **fail / abort** before the fix, while the P2P cases pass.
4. Apply `solution/code.patch` and **rebuild**.
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

Selected nodes in `test/dygraph_to_static/test_build_strategy.py`:

| Node | Role | Before fix | After fix |
| --- | --- | --- | --- |
| `TestResnetWithPass::test_resnet__ast_legacy_ir` | P2P | pass | pass |
| `TestError::test_type_error__ast_legacy_ir` | P2P | pass | pass |
| `TestError::test_type_error__sot_legacy_ir` | P2P | pass | pass |
| `TestResnetWithPass::test_resnet__ast_pir_exe` | F2P | abort (missing PIR op) | pass |
| `TestResnetWithPass::test_in_static_mode_mkldnn__ast_pir_exe` | F2P | abort (missing PIR op) | pass |

Every block in `tests/test.sh` runs even if an earlier one fails, and the script exits non-zero if any block fails. Each F2P node runs in its own pytest process because the base revision aborts natively on these nodes, which would otherwise drop the remaining results.

## Excluded Nodes

- Module-level `test_ast_only` / `test_legacy_and_pir`: these are decorators imported from `dygraph_to_static_utils_new`, not tests. Collecting the whole file makes pytest treat them as tests and error on the missing `fn` fixture. `tests/test.sh` selects classes / generated test ids instead, so they are not collected.
- `TestResnetWithPass::test_in_static_mode_mkldnn__ast_legacy_ir`: the legacy-IR oneDNN case already exists at the base revision and is not touched by this task. On the verification CPU it aborts (SIGABRT) in both the base and gold runs: oneDNN rejects the max-pooling backward primitive descriptor as unimplemented because the gradient memory layout does not match the forward hint. It is neither a valid F2P nor P2P, and running it in the same process as the other nodes would kill the whole run.

## Known Risks

- Build strategy / ResNet style cases can be heavier than unit op tests; prefer stable nodeids when selecting F2P/P2P.
- Historical PIR stack needs era-matched source build.
- Do not switch back to collecting the whole test file: it reintroduces the helper collection errors and the legacy-IR oneDNN abort.
