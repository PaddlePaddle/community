# Python 2.7.18 LoongArch64 移植构建步骤

> 完整构建文档，含双架构对比（x86_64 / loongarch64）

---

## 1. 环境要求

### 构建主机

| 项目 | 值 |
|---|---|
| 架构 | `loongarch64` |
| 操作系统 | AOSC OS 13.2.0 (Meow) |
| 内核 | Linux 7.0.11-aosc-main-4k |
| GCC | 12.3.0 20230508 (Anolis OS 12.3.0-17) |
| GLibc | 2.38 |

### 构建依赖

```
BuildRequires: gcc, make, glibc-devel
BuildRequires: zlib-devel, bzip2-devel, xz-devel
BuildRequires: openssl-devel, libffi-devel
BuildRequires: readline-devel, ncurses-devel
BuildRequires: sqlite-devel, gdbm-devel
BuildRequires: tk-devel, tcl-devel
BuildRequires: expat-devel
BuildRequires: pkgconfig
```

安装命令（AOSC / Anolis）:

```bash
sudo dnf install -y gcc make glibc-devel zlib-devel bzip2-devel \
    xz-devel openssl-devel libffi-devel readline-devel \
    ncurses-devel sqlite-devel gdbm-devel tk-devel tcl-devel \
    expat-devel pkgconfig
```

---

## 2. 源码准备

```bash
cd ~/src/python2.7-port
tar xf Python-2.7.18.tar.xz
cd Python-2.7.18
```

---

## 3. Patch 应用

```bash
# Patch 1: config.guess/config.sub 更新（识别 loongarch64）
patch -p1 -b .config-guess-orig < ../rpmbuild/SOURCES/python2.7-loongarch64-config-guess.patch

# Patch 2: OpenSSL 3.x 兼容（_hashlib 模块版本检测）
patch -p1 -b .openssl3-orig < ../rpmbuild/SOURCES/python2.7-openssl3-hashlib.patch
```

---

## 4. 替换 config.guess / config.sub

Python 2.7.18 自带的是 2017 年版本，无法识别 loongarch64。
需要替换为系统提供的 LA64 感知版本：

```bash
cp /usr/lib/rpm/anolis/config.guess config.guess
cp /usr/lib/rpm/anolis/config.sub config.sub
```

---

## 5. 配置

```bash
./configure \
    --with-system-ffi \
    --enable-shared \
    --with-threads \
    --with-signal-module \
    --with-wctype-functions \
    --enable-ipv6 \
    --with-system-expat \
    --with-dbmliborder=gdbm:bdb
```

关键选项说明：

| 选项 | 说明 |
|---|---|
| `--with-system-ffi` | 使用系统 libffi（3.4.4），而非捆绑的过时版本（2013, 不支持 LA64）|
| `--enable-shared` | 构建 libpython2.7.so |
| `--with-system-expat` | 使用系统 expat |
| `--with-dbmliborder=gdbm:bdb` | GDBM 优先于 Berkeley DB |

---

## 6. 编译

```bash
make -j$(nproc)
```

预期输出：`libpython2.7.so.1.0` + python 二进制 + 所有标准库模块。

---

## 7. 验证（Smoke Test）

```bash
PYTHONHOME=$PWD LD_LIBRARY_PATH=$PWD ./python -c "
import sys
assert sys.version.startswith('2.7.18')
import _ctypes, hashlib, ssl, sqlite3, readline, zlib, bz2, gdbm
print('Python 2.7 smoke test: PASS')
print('OpenSSL:', ssl.OPENSSL_VERSION)
print('SQLite:', sqlite3.sqlite_version)
"
```

---

## 8. RPM 打包

```bash
cd ~/src/python2.7-port/rpmbuild
rpmbuild -ba --define "_topdir $(pwd)" SPECS/python2.spec
```

输出产物：

```
RPMS/loongarch64/python2-2.7.18-1.an23.loongarch64.rpm
RPMS/loongarch64/python2-libs-2.7.18-1.an23.loongarch64.rpm
RPMS/loongarch64/python2-devel-2.7.18-1.an23.loongarch64.rpm
RPMS/loongarch64/python2-debuginfo-2.7.18-1.an23.loongarch64.rpm
RPMS/loongarch64/python2-libs-debuginfo-2.7.18-1.an23.loongarch64.rpm
RPMS/loongarch64/python2-debugsource-2.7.18-1.an23.loongarch64.rpm
SRPMS/python2-2.7.18-1.an23.src.rpm
```

---

## 9. 双架构对比

| 项目 | x86_64 (参考) | loongarch64 (本移植) | 差异说明 |
|---|---|---|---|
| GCC 版本 | 12.x | 12.3.0 | 一致 |
| GLibc | 2.38 | 2.38 | 一致 |
| OpenSSL | 3.x | 3.6.2 | 均需 Patch |
| libffi | 3.4.4+ | 3.4.4 | 一致，无需捆绑 |
| config.guess | 系统自带 | 需替换 | LA64 特有 |
| Python 版本 | 2.7.18 | 2.7.18 | 一致 |
| 模块覆盖 | 完整 | 完整 | _ctypes, hashlib, ssl, sqlite3, readline, zlib, bz2, gdbm |
| 测试通过率 | ~78% | ~78% | 预期一致（Python 2.7 已 EOL，部分测试因环境淘汰而失败）|
| ENDIAN | little | little | 一致 |
| PAGESIZE | 4K | 4K / 16K / 64K | Python 自动检测 |
| 最大整型 | 2^63-1 | 2^63-1 | 均为 LP64 |

### 架构特有注意事项

1. **libffi**：Python 2.7 捆绑的 libffi（3.0.13, 2013）不支持 loongarch64，必须 `--with-system-ffi`。
2. **config.guess**：必须替换为 2022+ 版本，否则 `configure` 不认识 `loongarch64-*-linux*`。
3. **python-config**：spec 中需保留 `python2-config` 和 `python-2.7.pc` 两个别名，部分构建系统依赖。
4. **shebang 检测**：AOSC/Anolis 的 `brp-mangle-shebangs` 会拒掉 `#!/usr/bin/env python`，需在 spec 中 `%global __brp_mangle_shebangs %{nil}`。

---

## 10. 常见问题

### Q: configure 报错 "unrecognized option --with-system-ffi"

旧版本 Python 的 configure 不识别此选项。2.7.18 支持，如果不行请确认补丁正确。

### Q: make 时报错 openssl/opensslv.h not found

```bash
sudo dnf install openssl-devel
```

### Q: _hashlib 模块编译失败

确认 OpenSSL ≥ 3.0，且 `python2.7-openssl3-hashlib.patch` 已应用。
该补丁读取 `OPENSSL_VERSION_MAJOR` / `MINOR` / `PATCH` 宏（OpenSSL 3.x 新增）代替已废弃的 `OPENSSL_VERSION_NUMBER`。

### Q: rpmbuild 报错 "argument list too long"

`%files` 段中 `/usr/lib/python2.7/` 和 `%{_libdir}/python2.7/` 可能展开出大量文件。如果遇到此问题，检查 spec 中使用 `%exclude` 排除 `.pyc`/`.pyo`/test 目录减少条目数。
