# PaddlePaddle__Paddle-79321

This directory converts Paddle PR #79321 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79321](https://github.com/PaddlePaddle/Paddle/pull/79321) |
| PR title | `[API Compatibility] Return _IncompatibleKeys for set_state_dict` |
| Base commit | `3cb4059b8e870c818031779af94eae728177c2ac` |
| Gold/head commit | `d2427e1d53fbd4d65623af43fdaf26d76740feb3` |
| Merged at | `2026-06-18` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Make `paddle.nn.Layer.set_state_dict` return an object that preserves the existing two-value tuple behavior while also exposing `missing_keys` and `unexpected_keys` as named fields.

## F2P / P2P Matrix

| Case | Nodeid | Expected before solution | Expected after solution |
| --- | --- | --- | --- |
| F2P | `test/legacy_test/test_state_dict_convert.py::TestStateDictReturn::test_missing_keys_and_unexpected_keys_attr` | Fails because `set_state_dict` returns a plain tuple without named fields | Passes; named fields are available and share identity with indexed tuple elements |
| P2P | `test/legacy_test/test_state_dict_convert.py::TestStateDictReturn::test_missing_keys_and_unexpected_keys` | Passes with existing two-value tuple unpacking | Continues to pass with backward-compatible unpacking |

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR, limited to `python/paddle/nn/layer/layers.py`.
- `tests/test.patch`: regression test patch adding the named-field F2P while retaining the existing tuple-unpacking P2P.
- `tests/test.sh`: minimal target test command for the F2P and P2P nodeids.
- `environment/README.md`: environment notes for static preparation and verifier runtime responsibility.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should leave the existing P2P tuple-unpacking test passing and make the new named-field F2P fail; applying both `tests/test.patch` and `solution/code.patch` should pass both target tests.
