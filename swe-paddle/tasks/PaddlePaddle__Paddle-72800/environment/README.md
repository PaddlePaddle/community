# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `705356a392f8cbdf0571cd60f8c0462eab424a80`
- Gold commit: `beaa40bddb70ca079979c4a54f053aaf64d549ce`
- Resource: CPU
- GPU required: no
- Patch type: C++ kernel (CPU/GPU backends)
- Python dependencies: PaddlePaddle (source build), NumPy

The verifier should execute against the Paddle source revision represented by the selected patch state. A source build is required since the patch modifies C++ kernel code.

## Build Instructions

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Build Paddle from source (CPU-only build is sufficient):
   ```bash
   mkdir build && cd build
   cmake .. -DWITH_GPU=OFF -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```
4. Install the built Paddle package.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build and install Paddle from source.
3. Apply `tests/test.patch`.
4. Run the P2P tests; existing non-zero-size behavior should pass.
5. Run the 0-size tensor tests; the target case should fail before the fix.
6. Apply `solution/code.patch`.
7. Rebuild Paddle from source.
8. Reinstall Paddle package.
9. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | cummin/cummax F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
