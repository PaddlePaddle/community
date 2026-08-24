# PaddlePaddle__Paddle-79276

This directory converts Paddle PR #79276 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79276](https://github.com/PaddlePaddle/Paddle/pull/79276) |
| PR title | `fix add_n 0-size bug` |
| Base commit | `8cacdfd15bc89296682c784df5b1685a7ca6e408` |
| Gold/head commit | `fa6bfed3dde252b97c9db8e32ce4d8bdd813b8a4` |
| Merged at | `2026-06-09` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

`paddle.add_n` should reject a list that mixes a zero-size tensor with a non-zero-size tensor of a different shape. The bug is that the zero-size shape can be treated as if no reference shape had been selected, so the later non-zero-size shape is accepted instead of raising the normal shape mismatch error.

## Why This Is A Good Candidate

- It is a real framework-level bug from a merged Paddle pull request.
- The failure is externally observable through the public `paddle.add_n` API.
- The target test is small, deterministic, CPU-executed, and does not require CUDA, custom devices, network access, model files, or random data.
- A correct fix must preserve valid all-zero-size inputs and ordinary compatible inputs while rejecting incompatible mixed-size inputs.
- The gold patch is a focused production change in C++ infermeta; real validation requires a rebuilt or otherwise equivalent compiled Paddle runtime.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: behavioral regression tests for `test/legacy_test/test_add_n_op.py`.
- `tests/test.sh`: minimal target pytest command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected matrix:

| Patch state | P2P guards | F2P target |
| --- | --- | --- |
| Base commit + `tests/test.patch` | PASS | FAIL |
| Base commit + `tests/test.patch` + `solution/code.patch` + rebuilt/equivalent compiled runtime | PASS | PASS |

Runtime Run/Test/Fix validation requires applying the patches to the target Paddle checkout and using a rebuilt or equivalent compiled runtime; that validation is the SWE-Paddle verifier's responsibility.
