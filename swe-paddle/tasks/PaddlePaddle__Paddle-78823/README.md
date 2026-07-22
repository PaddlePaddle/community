# PaddlePaddle__Paddle-78823

This directory converts merged Paddle PR #78823 into a SWE-Paddle community task package.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [78823](https://github.com/PaddlePaddle/Paddle/pull/78823) |
| PR title | `[API Compatibility] add pin_memory for randint` |
| Base commit | `c3a5e799eb9e390f830c9e6c3fbea2e9370afa7f` |
| Gold commit | `ddb483237539d4d23c7dbbd44e3a360439c780ed` |
| Merged at | `2026-05-12` |
| Task type | `bug_fix` |
| Track | Python API compatibility |
| Resource | CUDA or XPU for complete verification; CPU-only for limited error-path coverage |

## Summary

Make `pin_memory=True` use consistent place-conversion semantics across tensor creation, random generation (including `randint`), and audio window creation. CPU placement must use the available CUDA/XPU pinned allocator, explicit accelerator and already-pinned places must resolve consistently, and builds without a pinned allocator must report an explicit unsupported error.

## Scope

The gold change is Python-only and covers place handling used by tensor creation and random APIs. The upstream regression tests exercise CUDA, XPU, already-pinned, CPU fallback, and unsupported CPU-only behavior; hardware-dependent cases remain conditional.

## Artifacts

- `proposal.md`: original candidate proposal and source rationale.
- `instruction.md`: self-contained task requirements for the coding agent.
- `environment/README.md`: checkout, build, device, and reproduction notes.
- `solution/code.patch`: exact production-file diff from base to gold commit.
- `tests/test.patch`: exact test-file diff from base to gold commit.
- `tests/test.sh`: strict wrapper for the five target test files.

## Verification

From a built Paddle checkout with the patches applied in the documented order:

```bash
bash tests/test.sh
```

A CPU-only run is not complete verification because accelerator allocator branches are unavailable or skipped.
