# Environment Notes

This candidate is a Windows-specific SWE-Paddle task for Paddle Inference path handling.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `fcf3b100085b10efed4c1fb8880b1df1fd5241d6`
- Resource: Windows CPU
- GPU required: no
- Compiler: MSVC x64
- Recommended generator: Ninja
- Build path: Paddle source checkout at the base commit. This task targets a C++ Inference API helper, so a source build or equivalent compiled environment is required.

Linux-only verification is insufficient because the target bug is in Windows filesystem path handling.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Configure a Windows CPU build with the target-only switch enabled.
4. Build the target C++ test.
5. Run `bash tests/test.sh`; the UTF-8 path behavior should fail before the fix while the ASCII-path guard should pass.
6. Apply `solution/code.patch`.
7. Rebuild the target C++ test.
8. Run `bash tests/test.sh` again; the target tests should pass after the gold patch.

## Suggested Build Command

```bash
cmake -S . -B build   -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_C_COMPILER=cl.exe   -DCMAKE_CXX_COMPILER=cl.exe   -DPADDLE_INFERENCE_UTF8_PATH_TEST_ONLY:BOOL=ON   -DWITH_TESTING:BOOL=ON   -DWITH_INFERENCE_API_TEST:BOOL=OFF   -DON_INFER:BOOL=ON
```

Build the target:

```bash
cmake --build build --target inference_api_path_test --parallel 2
```

Some Windows builds may require preparing third-party headers before the target build:

```bash
cmake --build build --target extern_dirent --parallel 1
```

## Minimal Test Command

```bash
bash tests/test.sh
```

By default, the test script runs both target CTest cases. A verifier may set `TEST_REGEX` to run P2P and F2P separately, for example:

```bash
TEST_REGEX='^inference_api_path_p2p_test$' bash tests/test.sh
TEST_REGEX='^inference_api_utf8_path_f2p_test$' bash tests/test.sh
```

## Expected Results

| State | P2P | F2P |
| --- | --- | --- |
| `base + tests/test.patch` | PASS | FAIL |
| `base + tests/test.patch + solution/code.patch` | PASS | PASS |
