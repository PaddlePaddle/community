# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `6ad407d1aaf34c41e193361d22f10c39946b715f`
- Gold commit: `81a61590467551723341382599b950b4e92a6e1e`
- Resource: CPU
- GPU required: no
- Patch type: C++ kernel (phi kernels) + Python test
- Python dependencies: PaddlePaddle (source build), NumPy, pytest

The verifier should execute against the Paddle source revision represented by the selected patch state. Since the patch modifies C++ kernel files, a source rebuild is required after applying the solution patch.

## Build Instructions

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build Paddle from source (CPU-only build is sufficient):
```bash
mkdir build && cd build
cmake .. -DWITH_GPU=OFF -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
3. Install the built Paddle package.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build and install Paddle from source.
3. Apply `tests/test.patch`.
4. Run the P2P tests; existing non-zero-size behavior should pass.
5. Run the 0-size tensor tests; the target cases should fail before the fix.
6. Apply `solution/code.patch`.
7. Rebuild Paddle from source so the modified C++ kernels take effect:
```bash
cd build
make -j$(nproc)
pip install --force-reinstall python/dist/paddlepaddle-*.whl
cd ..
```
8. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | instance_norm F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
