# PaddlePaddle__Paddle-79391

This directory converts Paddle PR #79391 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79391](https://github.com/PaddlePaddle/Paddle/pull/79391) |
| PR title | `[API Compatibility] enhance paddle.enable_compat -part` |
| Base commit | `ad1d2d4df4731d62fe41263e276bb7d7f30e16e7` |
| Gold commit | `718431cf276c9bd32c089ee2daf6cc7d54af2aa8` |
| Merged at | `2026-07-22T05:08:19Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add an explicit `level=2` to `paddle.enable_compat()` that routes `paddle.*` and `paddle.Tensor` methods to
the torch-aligned `paddle.compat.*` implementations for external callers, while Paddle's own internals keep
native semantics — and register the `paddle.compat` root package with the torch import proxy so the
top-level compat functions are reachable at both levels.

## Why This Is A Good SWE-Paddle Candidate

- Real API-compatibility work: it is what makes prefix-only PyTorch→Paddle conversion (`torch.sort(x, dim=-1)`
  → `paddle.sort(x, dim=-1)`) actually run, without changing the meaning of existing `enable_compat()` calls.
- Non-trivial by construction. `paddle.X` and `paddle.compat.X` take the same types in the same positions, so
  no per-call type dispatch can tell the two contracts apart; the solution has to be a caller-aware process
  switch, not a `setattr` alias.
- Broad but bounded surface: `sys.meta_path` import hooks, runtime namespace rewriting, caller-aware
  dispatch, class proxies and metaclasses, `paddle.Tensor` method binding, exact save/restore of global
  state, and Paddle's own composite operators.
- Sharp boundaries that are easy to get wrong: `level=1` must stay byte-identical in behaviour; only public
  symbols that *already exist* on the paddle side may be taken over, so compat-only names
  (`slogdet`, `AvgPool1d`, `MultiheadAttention`, …) must not leak into the paddle namespace; several compat
  implementations call back into the same native API and must not recurse.
- Deterministic and CPU-only. No GPU, dataset, network, or distributed topology.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR (`python/paddle/compat/**`, pure Python).
- `tests/test.patch`: exact upstream test changes exposing the target behavior.
- `tests/test.sh`: minimal target test command, P2P before F2P, one interpreter per module.
- `environment/README.md`: environment notes, run order, and isolation requirements.
- `README.md`: task overview and verification entrypoint.

## Test roles

| Role | Module | Base | Base + gold patch |
| --- | --- | --- | --- |
| P2P | `test/compat/test_torch_proxy.py` | 12 passed | 12 passed |
| P2P | `test/compat/test_torch_proxy_mixed.py` | 4 passed | 4 passed |
| F2P | `test/compat/test_compat_namespace_aliased.py` | collection error: no module `paddle.compat.api_dispatch` | 29 passed, 14 subtests |
| F2P | `test/compat/test_compat_level2_internal_composites.py` | 10 errors: `enable_compat()` has no `level` | 10 passed |

Each module runs in its own interpreter — they mutate `sys.meta_path`, `sys.modules["torch*"]`, `sys.path`
and, under level 2, the `paddle` namespace itself. The torch-proxy P2P modules additionally require an
interpreter where PyTorch is **not** installed, which is what upstream Paddle CI provides; see
`environment/README.md`.

## Verification

```bash
PYTHON_BIN=python bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` keeps both torch-proxy P2P modules passing
while both new F2P modules fail (exit `1`); applying `solution/code.patch` on top makes all four modules
pass (exit `0`). The gold patch is pure Python, so no rebuild is needed — only a `paddle` package that is
actually loaded from the patched tree.
