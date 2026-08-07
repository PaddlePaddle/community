# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ac82c42a5c17f1ddd3ac50a28bb8d0ce84acba8e`
- Gold commit: `ea80fa17d84889795799fa5f868572b24bd8837c`
- Resource: CPU
- GPU required: no
- Patch type: Python API (no C++ changes, no rebuild required)
- Python dependencies: PaddlePaddle (pre-built or source), NumPy

The verifier should execute against the Paddle source revision represented by the selected patch state. Since the patch only modifies Python code, a pre-built Paddle installation with the patched Python files is sufficient.

## Setup Instructions

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Ensure Paddle is installed (pre-built wheel or source build).
4. Run the test commands.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Ensure Paddle is installed.
3. Apply `tests/test.patch`.
4. Run the P2P tests; existing non-zero-size behavior should pass.
5. Run the 0-size tensor tests; the target cases should fail before the fix.
6. Apply `solution/code.patch`.
7. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | pinv F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
