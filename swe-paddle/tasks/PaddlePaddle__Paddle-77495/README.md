# PaddlePaddle__Paddle-77495

This directory converts Paddle PR #77495 into a SWE-Paddle community task.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [#77495](https://github.com/PaddlePaddle/Paddle/pull/77495) |
| PR title | 【Hackathon 10th Spring No.1】Add `dilation` option for MaxPool1D/2D/3D -part |
| Base commit | `0604f65af5397848b6803c2bf577b9b82b8d8e08` |
| Merged at | `2026-02-06T02:59:55Z` |
| Merge commit | `3cc3127674c55ceb8c8d24b15b4c2e6504066d0a` |
| Task type | `feature_enhancement` |
| Resource | Verified with single GPU |

## Summary

Add dilated max pooling to the functional and layer forms of
`MaxPool1D`, `MaxPool2D`, and `MaxPool3D`, including shape inference,
forward kernels, backward kernels, and indexed pooling.

## Gold Patch Scope

The upstream PR branch merged `develop` before it was merged. To avoid
including unrelated changes, `solution/code.patch` is based on the PR's
feature commit chain through `6c313e19f246999322cf954d9560757624eec3b8`,
before that merge.

The gold patch also includes the narrowly scoped production fix from
[PR #77789](https://github.com/PaddlePaddle/Paddle/pull/77789). That follow-up
was required after compatibility testing of #77495. Changes from the related
#77681 PR that do not concern `MaxPool1D/2D/3D` were excluded.

Upstream tests are excluded from the gold patch. The independent test patch
provides a smaller CPU verifier.

## Upstream Review Findings

The package scope and verifier were refined from the review history of
[#77495](https://github.com/PaddlePaddle/Paddle/pull/77495) and
[#77789](https://github.com/PaddlePaddle/Paddle/pull/77789):

- Non-default dilation is dispatched through the indexed max-pool kernels,
  while the default path must preserve existing pooling behavior.
- A partially non-default vector such as `[1, 2]` must still select the
  dilation-aware branch.
- Reviewers required numerical backward validation rather than shape-only
  checks; the verifier therefore asserts exact gradient placement.
- CPU/GPU tests should follow the active CI device instead of forcing a device.
  The minimum verifier remains CPU-only, while the production patch retains
  CUDA support.
- XPU's underlying SDK did not support this feature. The accepted behavior is
  an explicit unsupported error, and the upstream test was excluded through
  CMake rather than repeated per-test skips.
- Post-merge compatibility testing found a public-call regression. The
  follow-up review required both pure positional conventions for all 1D/2D/3D
  functional and layer APIs, and noted that functional and layer APIs do not
  order their final boolean parameters identically.

The last item is intentionally represented only in hidden verifier coverage,
not explained in `instruction.md`.

| Review evidence | Link |
| --- | --- |
| Preserve the default dispatch path | [#77495 review](https://github.com/PaddlePaddle/Paddle/pull/77495#discussion_r2727358151) |
| Handle partially non-default dilation vectors | [#77495 review](https://github.com/PaddlePaddle/Paddle/pull/77495#discussion_r2740495401) |
| Validate backward values with OpTest | [#77495 review](https://github.com/PaddlePaddle/Paddle/pull/77495#discussion_r2740569842) |
| Avoid repeated XPU test skips | [#77495 review](https://github.com/PaddlePaddle/Paddle/pull/77495#discussion_r2762924875) |
| Compatibility regression report | [#77495 follow-up](https://github.com/PaddlePaddle/Paddle/pull/77495#issuecomment-3870010223) |
| Test both pure positional conventions | [#77789 review](https://github.com/PaddlePaddle/Paddle/pull/77789#discussion_r2786716168) |
| Functional/layer positional orders differ | [#77789 review](https://github.com/PaddlePaddle/Paddle/pull/77789#discussion_r2787314622) |

## Test Curation

The verifier separates baseline guards from target behavior:

| Class | Coverage |
| --- | --- |
| P2P | Existing functional/layer positional calls remain valid in 1D/2D/3D |
| F2P core | Functional and layer APIs for 1D, 2D, and 3D dilation |
| F2P kernel | CPU values, exact gradients, indexed 2D/3D OpTest nodes, asymmetric dilation, static graph, mask, `ceil_mode`, and layer repr |
| F2P compatibility | Both pure positional conventions and keyword aliases across all six public APIs |

These node groups can also serve as graded acceptance levels, but all must pass
for a fully verified instance. The compatibility trap and implementation
strategy are intentionally kept out of `instruction.md`.

### Reviewer-only compatibility matrix

For each of 1D, 2D, and 3D, the hidden contract compares the output and mask
from all rows below:

| Surface | Call shape after `padding` |
| --- | --- |
| Functional, established Paddle order | `return_mask, ceil_mode, dilation` |
| Functional, compatible order | `dilation, ceil_mode, return_indices` |
| Layer, established Paddle order | `return_mask, ceil_mode, dilation` |
| Layer, compatible order | `dilation, return_indices, ceil_mode` |
| Keyword aliases | `input=...` and `return_indices=True` |

This distinction is deliberate: using one positional remapping for both the
functional and layer APIs is one of the failure modes identified during
#77789 review.

## Files

- `proposal.md`: approved proposal and reviewer context.
- `instruction.md`: self-contained coding-agent problem statement.
- `solution/code.patch`: production-only gold patch.
- `tests/test.patch`: independent CPU contract tests plus focused upstream
  OpTest-wrapper and layer-repr updates.
- `tests/test.sh`: minimal test command.
- `environment/README.md`: build and Run/Test/Fix notes.

## Verification

```bash
TASK_DIR=/path/to/community/swe-paddle/tasks/PaddlePaddle__Paddle-77495
bash "$TASK_DIR/tests/test.sh"
```

At the base commit, applying only `tests/test.patch` must fail. After applying
`solution/code.patch` and rebuilding Paddle, all target tests must pass.
