# 杂项记录 / 踩坑笔记

> 构建和移植过程中遇到的问题、解决方案和各种备忘

---

## 1. `#!/usr/bin/env python` shebang 被 brp-mangle-shebangs 拦截

**现象**: RPM 构建失败，报错 `brp-mangle-shebangs: mangling shebang in ...`

**原因**: AOSC/Anolis 的 `brp-mangle-shebangs` 脚本会检查所有脚本文件的 shebang，要求路径可执行。Python 2.7 大量使用 `#!/usr/bin/env python`，但构建时的系统 Python 是 3.x，`/usr/bin/env python` 指向 Python 3。

**解决**: 在 spec 中全局禁用：

```
%global __brp_mangle_shebangs %{nil}
```

---

## 2. Python 2.7 在构建时不可用

**现象**: 构建系统尝试使用 `python` 命令（Python 3）来预编译 `.pyc`，产生不兼容的字节码。

**原因**: AOSC 系统默认 Python 是 3.x，Python 2.x 不兼容 Python 3 的 `.pyc` 格式。

**解决**: 在 spec 中全局禁用字节码编译：

```
%global __brp_python_bytecompile %{nil}
```

---

## 3. 捆绑 libffi 过旧（2013 版），不支持 loongarch64

**现象**: `_ctypes` 模块编译失败，FFI 调用崩溃。

**原因**: Python 2.7.18 捆绑的 libffi 3.0.13（2013 年发行）不包含 loongarch64 支持。系统 libffi 3.4.4 已经原生支持。

**解决**: 使用 `--with-system-ffi` 配置选项。如果失败，检查是否安装了 `libffi-devel`。

**检测**: `_ctypes` 模块是否正确加载：

```python
import ctypes
print(ctypes.CDLL(None))
```

---

## 4. `config.guess` 不识别 `loongarch64`

**现象**:
```
configure: error: /bin/sh config.sub loongarch64-unknown-linux-gnu failed
```

**原因**: Python 2.7.18 自带的 autoconf 工具太老（2017）。

**解决**: 双重处理——先应用 Patch0（升级到 2022 版），再用系统 config.guess 覆盖。

---

## 5. OpenSSL 3.x 版本宏变更

**现象**: `_hashlib` 模块无法编译，但 `openssl-devel` 已安装。

**原因**: OpenSSL 3.0+ 弃用了 `OPENSSL_VERSION_NUMBER` 宏，改用 `OPENSSL_VERSION_MAJOR` / `MINOR` / `PATCH`。
Python 2.7.18 的 `setup.py` 只匹配了旧宏。

**解决**: 应用 `python2.7-openssl3-hashlib.patch`。

---

## 6. `sysconfig` 路径问题

`sysconfig.get_makefile_filename()` 可能因架构名称不同而找不到 Makefile。
Python 2.7 内部硬编码了 `$MACHDEP` 为 `linux2`，而系统 config.guess 可能输出 `linux-gnu`。
当前构建结果中 `sys.platform` 为 `linux2`，行为正确。

---

## 7. rpmbuild 的 `%files` 中注意分叉路径

Python 2.7.18 的 `make install` 会将库文件安装到两个位置：

- `/usr/lib/python2.7/` — 纯 Python 标准库
- `%{_libdir}/python2.7/` — 架构相关模块（如 `_ctypes.so`、`gdbm.so`）

在 loongarch64 上两者相同（lib == lib64），但在 spec 中仍需区分处理，
分别 `%exclude test/` 和 `%exclude *.pyc`。

---

## 8. `test` 目录体积

```
$ du -sh /usr/lib/python2.7/test/
≈ 30 MB
```

建议在 RPM 中排除（已通过 `%exclude` 实现），减少安装包体积约 60%。

---

## 9. `python-config` 别名

部分构建系统（如 meson、cmake）通过 `python2-config` 或 `python2.7-config` 查找 Python 2.7。
需要确保所有三个别名都存在：

- `python2.7-config`（原始名称）
- `python2-config`（通用名称）
- `python-config`（默认名称，可能冲突，需注意）

---

## 10. pkg-config 文件

Python 2.7.18 安装三个 `.pc` 文件，都需要包含在 devel 子包中：

- `python.pc` -> `python-2.7.pc`
- `python2.pc` -> `python-2.7.pc`
- `python-2.7.pc`（规范名称）

---

## TODO / 待办

- [ ] 运行完整 `python -m test.regrtest` 记录通过率（参考 x86_64 基线 ~78%）
- [ ] 检查 `test_ctypes` 是否全部通过（libffi 移植需要）
- [ ] 检查 `test_hashlib`、`test_ssl` 是否因 OpenSSL 3.x 有行为差异
- [ ] 在 x86_64 构建相同 spec 做对比测试
- [ ] 测试第三方 C 扩展（如 `cryptography`、`numpy`）能否正常编译
- [ ] 确认 `distutils` 能正确识别 loongarch64 并编译 C 扩展
