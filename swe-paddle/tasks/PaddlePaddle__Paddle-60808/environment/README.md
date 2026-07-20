# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `161551046fd4c2b8a4ce19eb50fd6f5f0eeb5645`
- Gold commit: `e2a324cb86120d2e82d9d9dbd9400d60e3a4bc8e`
- Resource: CPU
- Python dependencies: PaddlePaddle, NumPy, pytest
- Paddle rebuild required: no

The regression test imports the installed Paddle runtime and overlays only `broadcast_to` and `expand` from the checked-out `python/paddle/tensor/manipulation.py`. This avoids replacing the complete historical module while ensuring the selected revision controls the behavior under test.

The test selects the legacy static-graph API before importing Paddle because the affected Base implementation contains distinct dynamic, PIR, and legacy static paths.

## Run Order

1. Check out the Base commit.
2. Restore the exact Base blob for `python/paddle/tensor/manipulation.py`.
3. Apply `tests/test.patch`.
4. Run the integer-list P2P; it should pass.
5. Run the two Tensor-shape F2P tests; they should fail on Base.
6. Apply `solution/code.patch`.
7. Verify the target file blob matches the Gold commit.
8. Run `bash tests/test.sh`; all tests should pass.

## Minimal Command

```bash
bash tests/test.sh
```

## Expected Matrix

| State | P2P | 0-D Tensor F2P | 1-D shape Tensor F2P | Full script |
| --- | ---: | ---: | ---: | ---: |
| Base + tests | PASS | FAIL | FAIL | FAIL |
| Base + tests + solution | PASS | PASS | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
