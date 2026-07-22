# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `f4014bfa7b9acddfcfcaffb57b57b2a5c8fe9e7a`
- Gold commit: `bf24243bd9e640590362c3f92abd74ca69a3ef30`
- Resource: CPU
- Patch type: Python-only
- Python dependencies: pytest
- Paddle build required: no

The target module depends only on Python standard-library modules for these tests. The test file loads `python/paddle/framework/restricted_unpickler.py` directly from the checked-out repository, so validation does not depend on an installed Paddle wheel matching the historical revision.

## Run Order

1. Check out the repository at the base commit.
2. Apply `tests/test.patch`.
3. Run the P2P test; existing safe-class behavior should pass.
4. Run the inherited-hook tests; they should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | Inherited class F2P | Restricted-load F2P |
| --- | ---: | ---: | ---: |
| Base + test patch | PASS | FAIL | FAIL |
| Base + test patch + solution patch | PASS | PASS | PASS |

No GPU, distributed runtime, compiled Paddle extension, external service, or additional dataset is required.
