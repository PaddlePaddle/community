# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `3bfdd753fa54582b0a2ab6b47e4ea8092cec8187`
- Gold commit: `763c9f253350af557a781c27c40bc1a2b7b350d2`
- Resource: CPU
- GPU required: no
- Patch type: Python-only
- Python dependencies: PaddlePaddle, NumPy, pytest

The verifier should execute against the Paddle source revision represented by the selected patch state. A source build is not required when an equivalent Python overlay is available and the underlying runtime remains API-compatible.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run the P2P tests; existing error-handling behavior should pass.
4. Run the 0-size tensor tests; the target case should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | fold F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
