# PaddlePaddle__Paddle-60346

This directory converts Paddle PR #60346 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | `#58067` |
| PR | `#60346` |
| PR title | `【PIR API adaptor No.272】Migrate LookAhead to pir` |
| Base commit | `ac2de38a80dd6f2c66b4ef4ad515e209a0470fef` |
| Gold commit | `caa171552152424a2adcfe6bef9babb2039434a2` |
| PR head | `d8af262bd3e13695ae15c10c9419709bb13bc209` |
| Merged at | `2023-12-27T06:54:07Z` |
| Task type | `feature_enhancement` |
| Device | CPU |
| Native rebuild | No |

## Gold production scope

```text
python/paddle/incubate/optimizer/lookahead.py
python/paddle/optimizer/sgd.py
```

The upstream PR also changes `test/legacy_test/test_lookahead.py`. That upstream test change is source evidence only; the SWE task uses an independent contract test under `test/swe_paddle/`.

## Test contract

`tests/test.patch` adds:

```text
test/swe_paddle/test_pr60346_lookahead_pir.py
```

The contract covers one P2P behavior and three F2P behaviors:

- legacy LookAhead global-step update remains valid;
- PIR LookAhead creates/updates its step without legacy static-only state construction;
- SGD accumulator creation accepts a PIR block;
- `LookAhead.minimize()` accepts a PIR value loss.

The tests are deterministic and CPU-only. They use AST extraction from the checkout source files so the verification is tied to the submitted source patch while avoiding build-tree/import-path coupling.

## Minimal verification

```bash
bash tests/test.sh
```

Expected matrix:

- Base: P2P passes, 3 F2P tests fail, full test command fails.
- Gold: all 4 tests pass.
