# PaddlePaddle__Paddle-78082

This directory packages Paddle PR #78082 as a SWE-Paddle task.

## Source Metadata

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| Pull request | [#78082](https://github.com/PaddlePaddle/Paddle/pull/78082) |
| Title | `[API Compatibility] add method pop(), values() and keys() to paddle.nn.ParameterDict` |
| Base commit | `ae907b878e91dbabf3582da99f8b05a46b588fc2` |
| Gold commit | `a2e4e5062dacbfef63cf4b08981b74b72ad21214` |
| Resource scope | CPU |
| Patch scope | Python API and legacy unittest coverage |

## Behavior Summary

The task adds PyTorch-compatible container operations to `paddle.nn.ParameterDict` without changing parameter registration or existing container behavior.

- `pop(key)` removes the named parameter and returns the same Parameter. A missing key raises `KeyError`.
- `keys()` exposes the current keys in insertion order, including order changes caused by `update()`.
- `values()` exposes the current Parameters in matching order. Returned values are Parameters, and the collection reflects removals.
- Existing indexing, iteration, update, forward/backward, registration, and state-dict behavior remains intact.

## Test Classification

- **F2P:** `TestParameterDictPopKeysValues` adds nine cases covering returned Parameters, missing keys, clearing by repeated pop, insertion order, update order, value shapes and types, value count, and synchronization after pop.
- **P2P:** `TestParameterDictStateDictRoundtrip` adds three state-dict roundtrip cases, while all pre-existing tests in `test_imperative_container_parameterdict.py` remain regression coverage.

## Artifact Map

- `proposal.md`: approved source proposal; preserved unchanged.
- `instruction.md`: self-contained observable requirements for the public container behavior.
- `solution/code.patch`: exact base-to-gold diff for the production file.
- `tests/test.patch`: exact base-to-gold diff for the legacy unittest file.
- `tests/test.sh`: strict direct-Python unittest wrapper.
- `environment/README.md`: revisions, runtime assumptions, patch order, and verification guidance.

## Verification Summary

- The production and test patches are exported from the exact base-to-gold Git diff with the requested one-file boundaries.
- Packaging checks cover patch whitespace, shell syntax, test-first application order, byte-for-byte gold equivalence, and Python syntax.
- Runtime verification is best-effort when the exact historical build is unavailable; the runtime commit and any compatibility limitation are recorded in `environment/README.md`.
