# 龙芯 LoongArch 架构软件包自动化移植使用手册

> **适用版本**: OpenClaw 移植方案 v0.1
> **目标读者**: LoongArch 生态开发者、发行版维护者
> **前置知识**: 基本的 Linux 命令行操作，了解 RPM 包管理

---

## 目录

1. [快速开始](#1-快速开始)
2. [环境准备](#2-环境准备)
3. [移植工作流](#3-移植工作流)
4. [常见场景操作](#4-常见场景操作)
5. [与 OpenClaw Agent 协作](#5-与-openclaw-agent-协作)
6. [结果验证与交付](#6-结果验证与交付)
7. [故障排查](#7-故障排查)
8. [文档导航](#8-文档导航)

---

## 1. 快速开始

### 1.1 一句话概述

本系统提供一套标准化的 LoongArch64 软件包移植方法：将 x86_64 上的 RPM 源码包，通过分类判定 → 参考构建 → 架构适配 → 测试验证 → RPM 交付的流水线，转化为 LoongArch64 可用的 RPM 包。

### 1.2 五分钟体验

如果你已经有一台 LoongArch64 机器并想快速体验，从一个最简单的 SRPM 开始：

```bash
# 1. 确认环境
uname -m                         # 应该输出 loongarch64
rpm --eval '%{_arch}'            # 应该输出 loongarch64

# 2. 安装构建工具
sudo dnf install -y rpmdevtools mock rpmlint
rpmdev-setuptree

# 3. 找一个已有的 SRPM（这里以 A 类包为例）
wget https://example.com/some-arch-independent-pkg.src.rpm

# 4. 在 LA64 上直接重建
mock --rebuild some-arch-independent-pkg.src.rpm -r loongarch64

# 5. 查看结果
ls /var/lib/mock/loongarch64/result/
```

如果成功，恭喜你完成了第一个 LA64 包的移植！

### 1.3 前置条件检查清单

开始前确认以下条件：

- [ ] LoongArch64 硬件或等效环境（QEMU 用户态模拟也可）
- [ ] 类 RHEL 发行版（Anolis OS / AOSC OS 等），已安装 RPM/DNF
- [ ] GCC ≥ 12.0（推荐 15.2.0）
- [ ] `rpmdevtools`、`mock`、`rpmlint` 已安装
- [ ] 目标软件的 x86_64 SRPM 或源码包已获取

---

## 2. 环境准备

### 2.1 安装构建工具链

```bash
# 基础工具
sudo dnf install -y rpm-build rpmdevtools rpmlint mock

# 编译工具链
sudo dnf install -y gcc gcc-c++ gcc-gfortran make cmake
sudo dnf install -y autoconf automake libtool pkgconfig

# 常用开发库
sudo dnf install -y glibc-devel zlib-devel bzip2-devel xz-devel
sudo dnf install -y openssl-devel libffi-devel

# 初始化 rpmbuild 工作目录
rpmdev-setuptree
```

### 2.2 配置 mock

创建 LoongArch64 的 mock 配置文件 `/etc/mock/loongarch64.cfg`：

```ini
config_opts['root'] = 'loongarch64'
config_opts['target_arch'] = 'loongarch64'
config_opts['legal_host_arches'] = ('loongarch64', 'x86_64')
config_opts['chroot_setup_cmd'] = 'install @buildsys-build'
config_opts['dist'] = 'el9'
config_opts['releasever'] = '9'

config_opts['dnf.conf'] = '''
[main]
keepcache=1
gpgcheck=0
assumeyes=1

[baseos]
name=Anolis OS $releasever - Base
baseurl=http://mirrors.example.com/anolis/$releasever/BaseOS/loongarch64/os/
enabled=1

[appstream]
name=Anolis OS $releasever - AppStream
baseurl=http://mirrors.example.com/anolis/$releasever/AppStream/loongarch64/os/
enabled=1
'''
```

> 将 `mirrors.example.com` 替换为实际的镜像站地址。

初始化 mock 环境：

```bash
mock --init -r loongarch64
```

### 2.3 添加 simde 支持（C 类包）

C 类包（使用 SIMD intrinsic）需要 SIMDE 可移植层：

```bash
sudo dnf install -y simde-devel
# 如果仓库中没有，从源码安装：
# git clone https://github.com/simd-everywhere/simde.git
# sudo cp -r simde/simde /usr/include/
```

### 2.4 配置 OpenClaw Agent (可选)

如果使用 Agent 辅助移植：

```bash
# 确保 Agent 可以读取项目文档
export OPENCLAW_DOCS=/path/to/docs-la-ports/src/agent-docs/
```

Agent 会在每次操作前加载 `src/agent-docs/` 中的行为规范和技术参考。

---

## 3. 移植工作流

### 3.1 完整流程图

```
获取源码 / SRPM
      │
      ▼
┌──────────┐
│ 阶段 0   │ 源码保护：git init + 初始提交
└──────────┘
      │
      ▼
┌──────────┐
│ 阶段 1   │ 信息收集：运行 classify-package.sh 分类
└──────────┘
      │
      ▼
┌──────────┐
│ 阶段 2   │ X86-64 参考构建：建立基线
└──────────┘
      │
      ▼
┌──────────┐
│ 阶段 3   │ LA64 适配：mock 构建 → 分析错误 → 修复
└──────────┘
      │
      ▼
┌──────────┐
│ 阶段 4   │ 测试验证 + 交付
└──────────┘
```

### 3.2 阶段 0：源码保护 — 必须执行

**这是不可跳过的步骤。**

```bash
cd /path/to/package-src
git init
git add -A
git commit -m "Initial import: <软件名>-<版本>"
```

为什么要做这一步：
- 任何移植修改都可以通过 `git diff` 追溯
- 可以随时回滚到原始状态
- 便于生成干净的移植补丁

### 3.3 阶段 1：信息收集与分类

#### 3.3.1 自动分类

运行分类脚本（位于本项目 `scripts/` 目录）：

```bash
bash /path/to/docs-la-ports/scripts/classify-package.sh package-1.2.3.tar.gz
# 输出: {"category": "B", "package": "package-1.2.3.tar.gz"}
```

分类结果含义：

| 类别 | 含义 | 你需要注意什么 |
|:---:|------|----------------|
| A | 架构无关 | 直接 mock --rebuild，无需修改源码 |
| B | 便携 C/C++ | 关注构建系统（autotools/cmake/meson），可能需要更新 config.guess |
| C | SIMD/Intrinsic | 需要安装 simde-devel，修改 `#include` 路径 |
| D | 内联汇编 | 需要逐段替换汇编代码，参考 `isa/ARCH_DIFF.md` |
| E | JIT/动态编译 | 最复杂，需要实现完整的代码生成后端 |

#### 3.3.2 手动判断

如果你不想用脚本，可以按以下顺序手动判断：

```
1. 项目是否包含任何 .c / .cpp / .f 文件？
   否 → 类别 A (架构无关)
   是 → 继续

2. 搜索内联汇编: grep -r "asm\s*(" src/
   有 → 类别 D

3. 搜索 SIMD: grep -r "mmintrin\|avx.*intrin\|arm_neon" src/
   有 → 类别 C

4. 搜索 JIT: grep -r "JIT\|jit_compil\|emit_code\|PROT_EXEC.*mmap" src/
   有 → 类别 E

5. 否则 → 类别 B
```

#### 3.3.3 提取构建信息

如果项目已有 spec 文件：

```bash
grep -E '^(Name|Version|BuildRequires|ExclusiveArch):' package.spec
```

如果没有 spec，查找构建系统文件：

```bash
ls -la configure.ac CMakeLists.txt meson.build Makefile 2>/dev/null
```

### 3.4 阶段 2：X86-64 参考构建

**在 x86_64 机器上执行**，建立构建基线。

```bash
# 1. 准备
cp package-1.2.3.tar.gz ~/rpmbuild/SOURCES/
cp package.spec ~/rpmbuild/SPECS/

# 2. 安装构建依赖
sudo dnf builddep ~/rpmbuild/SPECS/package.spec

# 3. 构建 SRPM (源码包)
rpmbuild -bs ~/rpmbuild/SPECS/package.spec

# 4. 在 mock 中构建二进制包（干净环境）
mock --rebuild ~/rpmbuild/SRPMS/package-*.src.rpm -r fedora-39-x86_64

# 5. 记录构建日志
cp /var/lib/mock/fedora-39-x86_64/result/build.log build-x86_64.log

# 6. 运行测试（如果有）
mock --shell -r fedora-39-x86_64 'cd /builddir/build/BUILD/package-1.2.3 && make test'
```

如果 x86_64 上构建失败，**先解决通用构建问题再移植**。

### 3.5 阶段 3：LoongArch64 适配

**在 LoongArch64 机器上执行**。

#### 3.5.1 首次构建尝试

```bash
# 使用 x86_64 产生的 SRPM 在 LA64 上构建
mock --rebuild package-1.2.3-1.el9.src.rpm -r loongarch64 2>&1 | tee build-la64-attempt1.log
```

#### 3.5.2 分析错误

根据错误类型查阅对应的解决指南：

| 错误信息关键字 | 查阅文档 | 常见解决方案 |
|----------------|----------|--------------|
| `cannot guess build type` | `la64-porting-guide.md` §1 | 替换 config.guess/config.sub |
| `unsupported architecture` | `la64-porting-guide.md` §1 | 同上，或修改 configure.ac |
| `fatal error: *mmintrin.h` | `la64-porting-guide.md` §2 | 用 SIMDE 头文件替换 |
| `implicit-function-declaration` | `la64-porting-guide.md` §3 | 添加 `-Wno-implicit-function-declaration` 或补充 include |
| `architecture is not included` | `la64-porting-guide.md` §9 | spec 的 `ExclusiveArch` 添加 `loongarch64` |
| `Failed build dependencies` | 依赖分析 | 检查 LA64 上是否有该依赖，若无则需先移植依赖 |

#### 3.5.3 应用修复

**A 类包**：通常只需要在 spec 中添加架构声明

```spec
# 在 spec 的 preamble 段添加
ExclusiveArch:  %{ix86} x86_64 aarch64 loongarch64
```

**B 类包**：处理构建系统 + 编译兼容性

```bash
# 1. 更新 autotools
cp /usr/share/autoconf/build-aux/config.guess ./
cp /usr/share/autoconf/build-aux/config.sub   ./

# 2. 在 spec 的 %build 段添加兼容标志
export CFLAGS="$CFLAGS -std=gnu89 -fcommon -Wno-error"
export CXXFLAGS="$CXXFLAGS -fpermissive"

# 3. 重新 build
autoreconf -fvis 2>/dev/null || true
./configure && make -j$(nproc)
```

**C 类包**：SIMDE 替换

```c
// 原代码
#include <immintrin.h>

// 修改为
#if defined(__loongarch__)
#  include <simde/x86/avx2.h>
#elif defined(__x86_64__)
#  include <immintrin.h>
#else
#  include <simde/x86/avx2.h>
#endif
```

并在 spec 中添加：

```spec
%ifarch loongarch64
BuildRequires:  simde-devel
%endif
```

**D 类包**：汇编替换，参考 `src/agent-docs/isa/ARCH_DIFF.md` 中的指令映射表。

**E 类包**：JIT 后端实现，参考 `docs/solutions/php-jit/` 的实战案例。

#### 3.5.4 迭代构建

```bash
# 修改源码后，重新生成 SRPM
rpmbuild -bs ~/rpmbuild/SPECS/package.spec

# 重新 mock 构建
mock --rebuild ~/rpmbuild/SRPMS/package-*.src.rpm -r loongarch64 2>&1 | tee build-la64-attempt2.log

# 循环直到成功
```

### 3.6 阶段 4：测试验证与交付

```bash
# 1. 安装构建产物
sudo dnf install /var/lib/mock/loongarch64/result/package-*.loongarch64.rpm

# 2. 功能验证
command --version
command --self-test

# 3. 运行测试套件
cd ~/rpmbuild/BUILD/package-1.2.3
make test 2>&1 | tee test-la64.log

# 4. 对比 x86_64 基线
diff <(grep PASS test-x86_64.log | sort) <(grep PASS test-la64.log | sort)

# 5. rpmlint 检查
rpmlint /var/lib/mock/loongarch64/result/package-*.rpm

# 6. 生成补丁
git diff > la64-adaptation.patch

# 7. 撰写移植记录
# 参考 docs/solutions/python2.7/ 的模板
```

---

## 4. 常见场景操作

### 4.1 场景 A：纯 Python / Perl / Shell 项目

```bash
# 这类项目架构无关，最简单 —— 直接 mock rebuild
mock --rebuild python-package-1.0-1.el9.src.rpm -r loongarch64

# 如果报错 "Architecture is not included: loongarch64"
# 在 spec 中添加一行：
# BuildArch: noarch
# 或
# ExclusiveArch: ... loongarch64
```

### 4.2 场景 B：标准的 autotools C 项目

```bash
# 步骤 1: 解压 SRPM 并修改
rpm -ihv package-1.0-1.el9.src.rpm
cd ~/rpmbuild

# 步骤 2: 编辑 SPECS/package.spec，在 %prep 段添加：
# %ifarch loongarch64
# cp -f %{_datadir}/autoconf/build-aux/config.guess ./config.guess
# cp -f %{_datadir}/autoconf/build-aux/config.sub   ./config.sub
# %endif

# 步骤 3: 重新构建
rpmbuild -bs SPECS/package.spec
mock --rebuild SRPMS/package-*.src.rpm -r loongarch64
```

### 4.3 场景 C：使用 CMake 的项目

CMake 项目通常对新架构支持较好。

```bash
# 1. 检查 CMakeLists.txt 中是否有架构检测
grep -i "arch\|x86\|arm\|aarch64" CMakeLists.txt

# 2. 在 spec 的 %build 段添加：
%cmake \
    -DCMAKE_BUILD_TYPE=Release \
    %ifarch loongarch64
    -DUSE_NATIVE_SIMD=OFF          # 禁用原生 SIMD
    -DSIMDE_INCLUDE_DIR=/usr/include/simde
    %endif

# 3. 构建
%cmake_build
```

### 4.4 场景：需要先移植依赖

如果构建时报 "Failed build dependencies"，说明 LA64 上缺少某个库：

```bash
# 1. 确认缺什么
dnf repoquery --whatprovides libfoo.so.1

# 2. 如果 LA64 仓库中不存在，就需要先移植该依赖
#    按照相同的流程移植 libfoo

# 3. 将移植好的 libfoo 安装到 mock 环境
mock --install /path/to/libfoo-*.loongarch64.rpm -r loongarch64

# 4. 继续构建主包
mock --rebuild package-*.src.rpm -r loongarch64
```

---

## 5. 与 OpenClaw Agent 协作

### 5.1 Agent 能做什么

OpenClaw Agent 是一个 AI 辅助移植工具，可以：

| 能力 | 自动程度 | 说明 |
|------|:---:|------|
| 分类判定 | 🟢 全自动 | 扫描源码特征，输出类别 |
| 信息收集 | 🟢 全自动 | 读取 spec/ebuild/PKGBUILD/CMakeLists |
| 参考构建 | 🟢 全自动 | 在 x86_64 执行 rpmbuild 并记录 |
| 错误分析 | 🟡 半自动 | 匹配构建错误到已知模式库 |
| 补丁生成 | 🟡 半自动 | 根据模式生成修复补丁 |
| spec 更新 | 🟢 全自动 | 在 spec 中插入 LA64 适配块 |
| 报告生成 | 🟢 全自动 | 生成 BUILD.md / TEST-REPORT.md |

### 5.2 如何使用 Agent

```bash
# 方式 1: 通过 Hermes Agent CLI 调用
hermes agent "请对 /path/to/package-1.2.3/ 进行 LoongArch64 移植分类判定"

# 方式 2: 直接与 Agent 对话
# 在 Agent 会话中：
"请将 ~/src/ffmpeg-6.1.1/ 移植到 LoongArch64，目标发行版是 Anolis OS 9"
```

Agent 会自动：
1. 读取 `src/agent-docs/README.md` 了解行为规范
2. 按照 5 阶段流程执行移植
3. 将结果记录到 `docs/solutions/<软件名>/`

### 5.3 Agent 行为的控制

Agent 在以下情况会**暂停并询问你**：

- 修改可能影响 x86_64 兼容性的代码
- 删除原始源码文件
- 需要跳过 x86_64 参考构建
- 发现没有构建文档的项目
- 遇到决策矩阵中的高风险操作

你可以随时说"继续"让 Agent 自主决策，或给出具体指示。

### 5.4 Agent 知识源

Agent 的知识来自以下文档：

```
src/agent-docs/
├── README.md                        ← 行为契约（必读）
├── business/domain-knowledge.md     ← 业务上下文
├── architecture/overview.md         ← 双架构环境
├── guidelines/
│   ├── la64-porting-guide.md        ← ★ 核心技术参考
│   ├── decision-matrix.md           ← 决策树
│   └── common-pitfalls.md           ← 历史教训
├── operations/safe-operations.md    ← 安全操作分级
└── isa/                             ← ISA 指令集模型
```

如果发现文档有误或不完整，Agent 会在操作中自行修正。

---

## 6. 结果验证与交付

### 6.1 验证检查清单

构建完成后，逐项检查：

```
□ 构建成功 (mock exit code = 0)
□ 无新增编译警告 (对比 x86_64 构建日志)
□ 二进制文件架构正确: file /path/to/binary | grep loongarch64
□ 依赖关系完整: rpm -qp --requires package-*.rpm
□ 基本功能可用: command --version 正常输出
□ 测试套件通过率 ≥ x86_64 基线
□ rpmlint 无严重错误
□ Git diff 可生成干净补丁
```

### 6.2 交付物

每次移植完成后应产出的文件：

```
docs/solutions/<软件名>/
├── BUILD.md           # 完整构建步骤（双架构对比）
├── PATCHES.md         # 补丁说明
├── TEST-REPORT.md     # 测试对比报告
├── NOTES.md           # 踩坑笔记
├── SOURCES/           # 补丁文件 (*.patch)
├── SPECS/             # 修改后的 spec 文件
├── SRPMS/             # 源码 RPM
└── RPMS/              # 二进制 RPM
    └── loongarch64/
```

### 6.3 提交到仓库

```bash
# 将 RPM 提交到本地仓库
cp RPMS/loongarch64/*.rpm /var/www/repo/loongarch64/
createrepo /var/www/repo/loongarch64/

# 或将补丁提交到上游
git format-patch HEAD~3    # 生成最近 3 个提交的补丁
```

---

## 7. 故障排查

### 7.1 构建失败速查表

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `Cannot guess build type` | config.guess 太旧 | 替换为系统版本 |
| `fatal error: *.h: No such file` | 缺少开发包 | `dnf install *-devel` |
| `undefined reference to ...` | 缺少链接库 | 添加 `-lfoo` 到 LDFLAGS |
| `error: Architecture is not included` | spec 不含 LA64 | 添加 `loongarch64` 到 `ExclusiveArch` |
| `Segmentation fault (core dumped)` | 汇编/JIT 指令错误 | 检查指令编码，对比 ISA 模型 |
| `mock: no configuration for loongarch64` | mock 配置缺失 | 创建 `/etc/mock/loongarch64.cfg` |
| `brp-mangle-shebangs` 报错 | shebang 不兼容 | 在 spec 中 `%global __brp_mangle_shebangs %{nil}` |

### 7.2 调试技巧

```bash
# 1. 进入 mock chroot 手动调试
mock --shell -r loongarch64
cd /builddir/build/BUILD/package-1.2.3
make V=1  # 显示完整编译命令

# 2. 查看 mock 构建日志
less /var/lib/mock/loongarch64/result/build.log

# 3. 验证二进制架构
file /var/lib/mock/loongarch64/result/usr/bin/some-binary
# 预期: ELF 64-bit LSB executable, LoongArch, ...

# 4. 使用 strace 调试运行时崩溃
mock --shell -r loongarch64
strace -f ./some-binary

# 5. GCC 扩展错误信息
make CFLAGS="$CFLAGS -v"  # 显示 GCC 调用的详细信息
```

### 7.3 常见命令参考

```bash
# ── 环境检查 ──
uname -m                          # CPU 架构
gcc -dumpmachine                  # GCC 目标平台
rpm --eval '%{_arch}'             # RPM 当前架构
ldd /path/to/binary               # 检查动态链接

# ── RPM 操作 ──
rpmdev-setuptree                  # 创建 rpmbuild 目录树
rpmbuild -ba package.spec         # 构建二进制 + 源码包
rpmbuild -bs package.spec         # 仅构建源码包
rpmlint package.spec              # 检查 spec 规范
rpmlint *.rpm                     # 检查包规范

# ── mock 操作 ──
mock --init -r loongarch64        # 初始化 chroot
mock --rebuild *.src.rpm -r loongarch64   # 从 SRPM 构建
mock --shell -r loongarch64       # 进入 chroot shell
mock --install *.rpm -r loongarch64       # 在 chroot 中安装包
mock --scrub=all -r loongarch64   # 清理 mock 环境

# ── 依赖管理 ──
dnf builddep package.spec         # 安装构建依赖
dnf provides '*/libfoo.so*'       # 查找哪个包提供某个文件
dnf repoquery --whatrequires foo  # 查找谁依赖 foo

# ── 源码分析 ──
grep -r 'asm\s*(' src/            # 搜索内联汇编
grep -r 'immintrin\|avx\|neon' src/  # 搜索 SIMD intrinsic
grep -r 'mmap.*PROT_EXEC' src/    # 搜索 JIT 特征
```

---

## 8. 文档导航

### 8.1 本文档体系的完整结构

```
docs-la-ports/
│
├── docs/                                    ← 方案与记录（实操导向）
│   ├── README.md                            # 入口
│   ├── loongarch-automated-porting-proposal.md  # 总体方案设计
│   ├── loongarch-porting-user-manual.md     # ★ 本文档：使用手册
│   ├── reports/
│   │   └── loongarch-porting-test-report.md # 测试报告
│   └── solutions/                           # 各软件包移植方案
│       ├── README.md                        # 方案索引
│       ├── python2.7/                       # Python 2.7.18
│       └── php-jit/                         # PHP 8.5.7 JIT
│
└── src/agent-docs/                          ← Agent 规范（技术参考）
    ├── README.md                            # Agent 行为契约
    ├── business/domain-knowledge.md         # 业务领域知识
    ├── architecture/overview.md             # 双架构环境
    ├── guidelines/
    │   ├── la64-porting-guide.md            # LA64 移植技术指南（核心）
    │   ├── decision-matrix.md               # 决策矩阵
    │   └── common-pitfalls.md               # 常见陷阱
    ├── operations/safe-operations.md        # 安全操作规范
    └── isa/                                 # ISA 指令集模型
```

### 8.2 我应该读哪些文档？

| 我想做什么 | 先读这个 | 再读这个 |
|------------|----------|----------|
| 快速上手移植一个包 | **本文档** (§1-§3) | `solutions/` 中的案例 |
| 了解整体方案设计 | `loongarch-automated-porting-proposal.md` | — |
| 处理具体的编译错误 | `src/agent-docs/guidelines/la64-porting-guide.md` | `solutions/<pkg>/NOTES.md` |
| 替换内联汇编 | `src/agent-docs/isa/ARCH_DIFF.md` | `src/agent-docs/isa/loongarch64/*.yml` |
| 知道什么风险不该自己做 | `src/agent-docs/operations/safe-operations.md` | `src/agent-docs/guidelines/decision-matrix.md` |
| 了解有哪些坑 | `src/agent-docs/guidelines/common-pitfalls.md` | `solutions/<pkg>/NOTES.md` |
| 使用 Agent 辅助 | **本文档** §5 | `src/agent-docs/README.md` |
| 查看测试结果 | `docs/reports/loongarch-porting-test-report.md` | `solutions/<pkg>/TEST-REPORT.md` |

---

> **文档维护**
>
> | 日期 | 版本 | 更新内容 | 更新者 |
> |------|------|----------|--------|
> | 2026-06-09 | v0.1 | 初版，覆盖环境准备、5阶段工作流、4类场景操作、Agent协作、故障排查 | Hermes Agent |
