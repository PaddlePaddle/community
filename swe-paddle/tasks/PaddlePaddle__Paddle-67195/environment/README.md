# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `87d69ba93e5db77d9c0647d5954bd43a7fcb5ea5`
- Gold commit: `d02476451b085229af521fa38d58a9f2a5ea1f9e`
- Resource: CPU
- Python dependencies: PaddlePaddle, pytest
- Paddle rebuild required: no

The verifier extracts `batch_send_recv_on_calc_stream` from the checked-out source file through Python AST and executes it with controlled Paddle and P2P communication doubles.

This validates the exact Base and Gold control flow while avoiding GPU, NCCL, process-group initialization, and network communication. The doubles record whether send or receive operations were invoked before an expected validation error.

## Run Order

1. Check out the Base commit.
2. Restore the exact Base blob for `p2p_communication.py`.
3. Apply `tests/test.patch`.
4. Run the valid-operation P2P; it should pass.
5. Run the single-invalid-send and mixed-batch F2P tests; both should fail on Base.
6. Apply `solution/code.patch`.
7. Verify the target file blob matches the Gold commit.
8. Run `bash tests/test.sh`; all verifier tests should pass.

## Minimal Command

```bash
bash tests/test.sh
```

## Expected Matrix

| State | Valid P2P | Invalid send F2P | Mixed batch F2P | Full script |
| --- | ---: | ---: | ---: | ---: |
| Base + tests | PASS | FAIL | FAIL | FAIL |
| Base + tests + solution | PASS | PASS | PASS | PASS |

No GPU, distributed launcher, process group, external service, or dataset is required.
