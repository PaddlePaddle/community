# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ae907b878e91dbabf3582da99f8b05a46b588fc2`
- Gold commit: `2d26799c4f1a710deb78b7a2182c60eaa0ee7d22`
- Resource: CPU
- GPU required: no
- Patch type: Python-only
- Python dependencies: PaddlePaddle, NumPy, pytest

The verifier should execute against the Paddle source revision represented by the selected patch state. A source build is not required when an equivalent Python overlay is available and the underlying runtime remains API-compatible.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run the P2P test; existing non-empty-index behavior should pass.
4. Run the zero-size tests; both target cases should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | Out-of-place F2P | In-place F2P |
| --- | ---: | ---: | ---: |
| Base + test patch | PASS | FAIL | FAIL |
| Base + test patch + solution patch | PASS | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
