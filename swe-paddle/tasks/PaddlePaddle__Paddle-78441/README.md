# PaddlePaddle__Paddle-78441

This directory converts merged Paddle PR #78441 into a SWE-Paddle community task package.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [78441](https://github.com/PaddlePaddle/Paddle/pull/78441) |
| PR title | `[API Compatibility] add aminmax op-part` |
| Base commit | `35b36cca24a780061268d20d6abe512e758837e6` |
| Gold commit | `156159726b64d8f85747de864fb3ce41ea1f3f2f` |
| Merged at | `2026-04-27` |
| Task type | `feature_implementation` |
| Track | Python API compatibility / new operator |
| Resource | CPU; source build required |

## Summary

Add `paddle.aminmax` and the corresponding Tensor method so one reduction returns both minimum and maximum values. The task covers the public API, two-output shape inference, CPU execution, gradients, static and dynamic graph behavior, compatibility aliases, output tensors, and symbolic shapes.

## Scope

The gold change includes operator schemas and code generation inputs, infermeta, CPU/GPU kernel registrations, backward support, PIR symbolic-shape inference, Python exports and signatures, plus focused tests. CPU is the benchmark acceptance backend; GPU registration is retained in the gold patch but does not make GPU hardware a verifier requirement.

## Test Classification

- **F2P:** the 26 cases in the new `test/legacy_test/test_aminmax_op.py` suite plus `test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py::AminmaxOpInferSymbolicShapeTest`, 27 nodes in total. They fail on the base build and pass after applying `solution/code.patch` and rebuilding. The symbolic-shape node only reports the `sym_shape_str` attribute it asserts on when it runs under the upstream FLAGS environment, which `tests/test.sh` now sets; see the verification notes below.
- **P2P:** four existing amin/amax regression nodes in `test/legacy_test/test_max_min_amax_amin_op.py`:
  - `TestAmaxAPI_Compatibility::test_dygraph_Compatibility`
  - `TestAminAPI_Compatibility::test_dygraph_Compatibility`
  - `TestAmaxAminOutAPI::test_amax_out_in_dygraph`
  - `TestAmaxAminOutAPI::test_amin_out_in_dygraph`

  The gold patch does not modify this file; these nodes must pass on both base and gold builds.

## Artifacts

- `proposal.md`: approved candidate proposal and source rationale.
- `instruction.md`: self-contained task requirements for the coding agent.
- `environment/README.md`: source-build and reproduction instructions.
- `solution/code.patch`: exact non-test diff from base to gold commit.
- `tests/test.patch`: exact test-only diff from base to gold commit.
- `tests/test.sh`: wrapper that sets the required per-suite `PYTHONPATH` and the symbolic-shape FLAGS environment, runs the four amin/amax P2P nodes, then runs the new legacy and symbolic-shape F2P targets.

## Verification

From the root of a Paddle checkout after applying the patches in the documented order:

```bash
bash tests/test.sh
```

On `base_commit + tests/test.patch`, the four P2P nodes must pass and both F2P targets must fail during API lookup, op creation, graph construction, or execution; the wrapper records both F2P failures and exits nonzero. On `base_commit + tests/test.patch + solution/code.patch` after rebuilding, all P2P and F2P targets must pass.

`tests/test.sh` sets `PYTHONPATH` per test suite:

- legacy operator tests need `test/legacy_test` and `test` on `PYTHONPATH`;
- the symbolic-shape test needs `test/ir/pir/cinn` before any directory containing the other `utils.py`, plus its own `test/ir/pir/cinn/symbolic` directory.

It also exports `FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1 FLAGS_prim_enable_dynamic=true FLAGS_prim_all=True FLAGS_cinn_new_group_scheduler=1` for the symbolic-shape node, mirroring how `test/ir/pir/cinn/symbolic/CMakeLists.txt` registers that file. Without those flags the node raises `KeyError: 'sym_shape_str'` on the gold build too, because the attribute it reads is only attached when `CheckInferSymbolicIfNeed` actually runs the shape optimization pass.

A pre-existing release or nightly wheel cannot validate this task because the change introduces a compiled operator and updates build-time code-generation inputs.
