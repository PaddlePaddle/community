# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `0a23433eddfd286cbdb8746240eaf662cd027c69`
- Gold commit: `1d3518f4ab0bb6f188e222152529f0b31f6acee3`
- Resource: CPU
- GPU required: no
- Patch type: Python-only
- Python dependencies: PaddlePaddle, NumPy, pytest

The verifier should execute against the Paddle source revision represented by the selected patch state. A source build is not required when an equivalent Python overlay is available and the underlying runtime remains API-compatible.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run the P2P tests; existing non-zero-size behavior should pass.
4. Run the 0-size tensor tests; the target case should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | dice_loss F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
