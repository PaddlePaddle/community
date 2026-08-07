# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `9c1900ce422e3398bfccf95d3d33ba2cfa91faed`
- Gold commit: `f8e6a83f6cb3725902082e7fbb011d7c1e8f6406`
- Resource: CPU
- GPU required: no
- Patch type: Python API (no C++ kernel changes)
- Python dependencies: PaddlePaddle (source build or pre-built), NumPy

The verifier should execute against the Paddle source revision represented by the selected patch state. Since the patch only modifies Python code, no recompilation is needed — only the Python files change.

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
4. Run the P2P tests; existing conv1d behavior should pass.
5. Run the CPU float16 test; the target case should fail before the fix.
6. Apply `solution/code.patch`.
7. Repackage and reinstall the Python files so the modified `conv.py` takes effect:
```bash
cd python
python setup.py bdist_wheel
pip install --force-reinstall dist/paddlepaddle-*.whl
cd ..
```
8. Run `bash tests/test.sh`; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Expected Matrix

| Revision state | P2P | conv1d F2P |
| --- | ---: | ---: |
| Base + test patch | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

No GPU, distributed runtime, external service, or additional dataset is required.
