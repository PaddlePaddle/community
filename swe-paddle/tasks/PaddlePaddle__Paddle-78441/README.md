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

## Artifacts

- `proposal.md`: approved candidate proposal and source rationale.
- `instruction.md`: self-contained task requirements for the coding agent.
- `environment/README.md`: source-build and reproduction instructions.
- `solution/code.patch`: exact non-test diff from base to gold commit.
- `tests/test.patch`: exact test-only diff from base to gold commit.
- `tests/test.sh`: strict target-test wrapper.

## Verification

From the root of a correctly rebuilt Paddle checkout, after applying the patches in the documented order:

```bash
bash tests/test.sh
```

A pre-existing release or nightly wheel cannot validate this task because the change introduces a compiled operator and updates build-time code-generation inputs.
