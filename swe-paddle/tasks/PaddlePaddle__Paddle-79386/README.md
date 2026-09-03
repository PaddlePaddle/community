# PaddlePaddle__Paddle-79386

This directory converts Paddle PR #79386 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79386](https://github.com/PaddlePaddle/Paddle/pull/79386) |
| PR title | `[API Compatibility] Bug Fix` |
| Base commit | `9aa3379edbee8ccd6cec772b22ad37733357f8df` |
| Gold commit | `d9b89b3918a51476cc1755fe202f89a07f8c34d1` |
| Merged at | `2026-06-29` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

Fix `paddle.iinfo(paddle.uint64).max` so Python receives the full unsigned 64-bit maximum value instead of a signed interpretation such as `-1`.

## Why This Is A Good Candidate

- It comes from a merged Paddle API-compatibility bug-fix PR.
- The production change is isolated to the C++/pybind exposure of integer dtype metadata.
- The target behavior is deterministic and covered by narrow dtype boundary assertions.
- It requires no GPU, distributed launch, external service, or network access.
- Real verification requires a source build or equivalent rebuilt `libpaddle`; an unchanged prebuilt wheel cannot exercise the patched C++ binding.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR.
- `tests/test.patch`: regression tests exposing the target behavior.
- `tests/test.sh`: minimal target test command for the dedicated class.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: after applying `tests/test.patch` to the base commit and running against the base runtime, the dedicated `uint64.max` F2P assertion should fail while the P2P guard assertions should pass. After also applying `solution/code.patch`, the verifier must rebuild Paddle from source or provide an equivalent rebuilt pybind runtime before rerunning `bash tests/test.sh`; the full dedicated class should then pass.

The verifier is responsible for performing the source build or equivalent runtime preparation and recording Run/Test/Fix results. This task cannot be verified with an unchanged prebuilt Paddle wheel.