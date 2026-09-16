# 移植补丁说明

> rpmbuild/SOURCES/ 目录下所有补丁的详细说明

---

## Patch 0: `python2.7-loongarch64-config-guess.patch`

**用途**: 将 Python 2.7.18 自带的 config.guess/config.sub 从 2017 版升级为 2022 版

**背景**: Python 2.7.18（2019 年发布）捆绑的 GNU config.guess（2017-05-27）和 config.sub 不包含 loongarch64 架构信息。没有这个补丁，`./configure` 会报错：

```
checking build system type... Invalid configuration `loongarch64-unknown-linux-gnu'
configure: error: /bin/sh config.sub loongarch64-unknown-linux-gnu failed
```

**变更概要**:
- config.guess: 2017-05-27 → 2022-05-25
- config.sub: 同步更新
- 新增 loongarch64* 的识别逻辑
- 修复 shebang 行（`#! /bin/sh` → `#!/usr/bin/sh`）

**注意事项**:
- RPM spec 中通过 `cp /usr/lib/rpm/anolis/config.guess` 进一步替换为系统提供的 LA64 优化版本
- 补丁仅提供基础识别能力，系统 config.guess 包含更多 LA64 变体（big-endian、软浮点等）

---

## Patch 1: `python2.7-openssl3-hashlib.patch`

**用途**: 修复 OpenSSL 3.x 下 `_hashlib` 模块的版本检测

**背景**: Python 2.7.18 默认使用 `OPENSSL_VERSION_NUMBER` 宏（形如 `0x1010103f`）判断 OpenSSL 版本。
OpenSSL 3.0+ **废弃**了此宏（值固定为 `0x30000000`），改用三个独立宏：

- `OPENSSL_VERSION_MAJOR` (3)
- `OPENSSL_VERSION_MINOR` (0)
- `OPENSSL_VERSION_PATCH` (0)

没有此补丁，`setup.py` 无法正确解析 OpenSSL 版本，导致 `_hashlib` 构建跳过（认为 OpenSSL 不可用或版本太旧）。

**变更概要** (`setup.py`):

```
- 仅匹配 OPENSSL_VERSION_NUMBER → 匹配所有四个宏
- 从 OPENSSL_VERSION_MAJOR / MINOR / PATCH 构建版本号
- 回退到 OPENSSL_VERSION_NUMBER（兼容旧版 OpenSSL 1.x）
```

**验证**:

```python
# 构建后测试
import hashlib
for algo in ['md5', 'sha1', 'sha256', 'sha512', 'blake2b', 'blake2s']:
    h = hashlib.new(algo)
    print(f'{algo}: {h.name}')
```

**对应 OpenSSL 版本**:

```
本系统 OpenSSL: 3.6.2 (7 Apr 2026)
补丁有效版本: OpenSSL 1.1.1+ 及 3.x 全系列
```

---

## 补丁应用顺序

```
Patch 0: config.guess/config.sub 升级
Patch 1: OpenSSL 3.x hashlib 兼容
```

必须先应用 config.guess 补丁，否则 configure 阶段就会失败。

---

## 未使用的补丁

### `python2.7-system-libffi.patch`

**状态**: 已废弃，保留仅供参考。

说明：此补丁原本用于强制使用系统 libffi（设置 `self.use_system_libffi = True`）。最终改用 `--with-system-ffi` configure 选项，该选项是 Python 2.7 官方支持的配置方式，无需 patch。

内容：

```diff
--- a/setup.py
+++ b/setup.py
@@ -2116,7 +2116,7 @@ class PyBuildExt(build_ext):
             return

     def detect_ctypes(self, inc_dirs, lib_dirs):
-        self.use_system_libffi = False
+        self.use_system_libffi = False  # controlled via --with-system-ffi configure flag
         include_dirs = []
         extra_compile_args = []
         extra_link_args = []
```

注意此 patch 仅改了注释，实际逻辑是 configure 时通过 `--with-system-ffi` 控制。
