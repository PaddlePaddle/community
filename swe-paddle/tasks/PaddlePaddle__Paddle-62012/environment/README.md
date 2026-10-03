# Environment Notes

## 预期环境

- 仓库：`PaddlePaddle/Paddle`
- Base commit：`1b68a51dbdc6b4e93a0c8e28df74e8d881272501`
- Python：3.11
- 运行时：Paddle CPU wheel
- GPU：不需要
- CUDA：不需要
- Torch：不需要
- 源码编译：不需要

## 执行顺序

1. 在 Base commit 上应用 `tests/test.patch`。
2. 运行 `bash tests/test.sh`，修复前应失败。
3. 应用 `solution/code.patch`。
4. 再次运行 `bash tests/test.sh`，修复后应通过。

## 最小测试命令

```bash
bash tests/test.sh
```

仓库根目录外的 `cross_PR62012.sh` 会自动设置 CPU、SOT 严格模式和源码覆盖路径，并保存完整日志。
