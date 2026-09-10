# 龙芯 LoongArch 架构软件包自动化移植方案文档

> **文档状态**: 初版 (v0.1)
> **适用范围**: 类 RHEL 发行版（OpenAnolis / Anolis OS）上的 LoongArch64 软件包移植
> **配套文档**: `src/agent-docs/` 下的 OpenClaw Agent 行为规范和技术参考

---

## 目录

1. [项目概述](#1-项目概述)
2. [目标架构与环境](#2-目标架构与环境)
3. [自动化移植体系架构](#3-自动化移植体系架构)
4. [软件包分类与移植策略](#4-软件包分类与移植策略)
5. [标准化移植流水线](#5-标准化移植流水线)
6. [构建基础设施](#6-构建基础设施)
7. [Agent 辅助自动化](#7-agent-辅助自动化)
8. [质量保证体系](#8-质量保证体系)
9. [移植数据库与知识沉淀](#9-移植数据库与知识沉淀)
10. [实施路线图](#10-实施路线图)

---

## 1. 项目概述

### 1.1 背景

LoongArch 是龙芯中科自主研发的 64 位指令集架构。随着龙芯 3A6000 等新一代处理器的发布，LoongArch 生态建设进入加速期。当前面临的核心挑战是：将 x86_64 / aarch64 生态中成熟的开源软件批量、高效地移植到 LoongArch64 平台。

传统移植方式依赖人工逐一处理，效率低下且经验难以复用。本方案提出一套 **自动化、标准化、可复用** 的软件包移植体系，结合 AI Agent（OpenClaw）辅助，实现从源码获取到 RPM 产出全流程的自动化。

### 1.2 目标

| 维度 | 目标 |
|------|------|
| **效率** | 将单个软件包的移植周期从人日均级降至小时级 |
| **质量** | 移植后的包通过相同的测试套件，行为与源架构一致 |
| **可复用** | 每个移植决策可记录、可回溯、可复现 |
| **规模化** | 支撑成百上千个软件包的批量移植 |
| **标准化** | 统一的 RPM spec 模板、构建流程、质量门禁 |

### 1.3 核心原则

1. **基线先行**: 任何移植必须先在 x86_64 上建立参考构建基线
2. **最小侵入**: 移植修改仅限于解决架构差异，不破坏原有架构兼容性
3. **可追溯**: 每处修改通过 Git 跟踪，每步决策有文档记录
4. **自动化优先**: 能脚本化的不手动作，能 Agent 处理的不人工介入
5. **知识沉淀**: 每完成一个移植，将经验结构化为可检索的数据

---

## 2. 目标架构与环境

### 2.1 硬件平台

```
CPU:       Loongson-3A6000 (LA664 微架构, 4 核 8 线程)
ISA:       LoongArch64 (LP64D ABI)
扩展:      LSX (128-bit SIMD), LASX (256-bit SIMD), LBT (二进制翻译辅助)
```

### 2.2 操作系统

```
目标发行版:  OpenAnolis / Anolis OS (类 RHEL)
包格式:      RPM
包管理器:    DNF / YUM
构建系统:    rpmbuild + mock
```

### 2.3 源架构（参考基线）

```
源架构:      x86_64 (Intel/AMD, 类 RHEL)
工具链:      GCC 15.x / Clang 19+
模拟:        QEMU user mode (x86_64 → loongarch64 交叉场景)
```

### 2.4 双架构环境模型

```
┌──────────────────────────────────────────────────────────────────┐
│                    OpenClaw 移植环境                              │
│                                                                  │
│  ┌─────────────────────┐        ┌──────────────────────────────┐ │
│  │   X86-64 环境        │        │  LoongArch64 环境             │ │
│  │  (阶段1-2)           │        │  (阶段3-4)                    │ │
│  │                      │        │                               │ │
│  │  - 信息收集          │  SRPM  │  - 接收 SRPM                 │ │
│  │  - X86 参考构建      │ ────→  │  - 架构适配                  │ │
│  │  - 生成 SRPM         │        │  - mock 隔离构建              │ │
│  │  - 记录基线日志      │  ←──── │  - 运行测试套件              │ │
│  │                      │  RPM   │  - 产出 LA64 RPM              │ │
│  └─────────────────────┘        └──────────────────────────────┘ │
│                                                                  │
│  交付物:                                                         │
│    - *.loongarch64.rpm     (二进制包)                             │
│    - *.src.rpm             (源码包，可复现)                        │
│    - *.patch               (架构适配补丁)                          │
│    - BUILD.md              (构建记录)                              │
│    - TEST-REPORT.md        (测试对比报告)                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 自动化移植体系架构

### 3.1 总体架构

```
                        ┌──────────────────┐
                        │   移植任务队列    │
                        │  (优先级排序)     │
                        └────────┬─────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    自动化移植引擎                                  │
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │  信息收集  │  │  参考构建  │  │  架构适配  │  │  测试验证  │     │
│  │  模块      │→ │  模块      │→ │  模块      │→ │  模块      │     │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    知识库 & 决策引擎                        │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐  │  │
│  │  │ ISA 模型 │ │ 移植模式库│ │ 构建失败DB │ │  测试基线DB  │  │  │
│  │  └─────────┘ └──────────┘ └────────────┘ └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │     交付物        │
                        │  RPM + Patch +     │
                        │  Build Log +       │
                        │  Test Report       │
                        └──────────────────┘
```

### 3.2 模块职责

| 模块 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 信息收集 | 源码 tarball / Git URL | 扫描构建系统、提取依赖、识别架构特性 | 构建计划 (JSON) |
| 参考构建 | 构建计划 | 在 x86_64 上执行构建，记录日志 | 基线构建日志 + SRPM |
| 架构适配 | SRPM + 构建日志 | 识别架构差异，应用修复策略 | 补丁集 + 修改后的 spec |
| 测试验证 | LA64 RPM | 运行测试套件，对比 x86_64 基线 | 测试报告 |

---

## 4. 软件包分类与移植策略

### 4.1 分类体系

按移植难度和技术特征将软件包分为 5 类：

| 类别 | 特征 | 典型代表 | 预估工作量 | 自动化程度 |
|------|------|----------|:---:|:---:|
| **A · 架构无关** | 纯脚本、纯数据、纯解释型语言 | Python/Perl 模块、字体、文档 | 分钟级 | 🟢 全自动 |
| **B · 便携 C/C++** | 标准 C/C++，无内联汇编，无 SIMD | zlib, bzip2, sqlite | 十分钟级 | 🟢 自动为主 |
| **C · 有 SIMD/Intrinsic** | 使用 SSE/AVX/NEON 等架构 intrinsic | ffmpeg, x264, opencv | 小时级 | 🟡 半自动 |
| **D · 内联汇编/手写汇编** | 直接使用 .asm / __asm__ | openssl, glibc, linux | 天级 | 🟠 需人工 |
| **E · JIT/动态编译** | 运行时生成机器码，含重定位/调用约定 | Python 3.14 JIT, V8, LuaJIT | 周级 | 🔴 深度人工 |

### 4.2 分类判定流程

```
源码包输入
    │
    ▼
┌─────────────────────────────┐
│ 是否含 .asm / __asm__ /     │
│ .S 汇编文件？               │──是──▶ 类别 D: 内联汇编
└─────────────────────────────┘
    │否
    ▼
┌─────────────────────────────┐
│ 是否含 xmm/emm/imm/avx      │
│ intrinsic 头文件？           │──是──▶ 类别 C: SIMD/Intrinsic
└─────────────────────────────┘
    │否
    ▼
┌─────────────────────────────┐
│ 是否涉及 JIT / 动态代码     │
│ 生成 / ELF 运行时重定位？    │──是──▶ 类别 E: JIT/动态编译
└─────────────────────────────┘
    │否
    ▼
┌─────────────────────────────┐
│ 是否含 C/C++/Fortran 代码？  │──是──▶ 类别 B: 便携 C/C++
└─────────────────────────────┘
    │否
    ▼
类别 A: 架构无关
```

### 4.3 各类别移植策略

#### 类别 A — 架构无关

```
策略: 直接在新架构上 rebuild，无需修改源码
操作:
  1. 从 x86_64 获取 SRPM
  2. 在 LA64 上 mock --rebuild
  3. 若成功 → 交付
  4. 若失败 → 检查 spec 中的 ExclusiveArch，添加 loongarch64
```

#### 类别 B — 便携 C/C++

```
策略: 解决构建系统对新架构的识别，无需修改业务代码
常见问题:
  - config.sub/guess 版本过旧 → 替换为支持 LA64 的版本
  - GCC/Clang 新版本语法严格 → 添加 -std=gnu89 -fcommon -Wno-error
  - glibc 内部宏变更 → -D__alloca=alloca -D__stat=stat
  - 隐式函数声明 → 补充 #include 或 -Wno-implicit-function-declaration
操作:
  1. 导入 spec文件
  2. 在 %prep 段添加 config.sub/guess 更新
  3. 在 %build 段添加兼容性编译标志
  4. mock 构建 → 测试 → 交付
```

#### 类别 C — SIMD/Intrinsic

```
策略: 使用 SIMDE 可移植模拟层，或添加 __loongarch__ 条件分支
常见场景:
  - X86 SSE/AVX intrinsic → 用 SIMDE 头文件替换
  - ARM NEON intrinsic → 用 SIMDE 头文件替换
  - 硬件加速（加密、视频编解码）→ 添加 LA64 LSX/LASX 实现或回退到 C
操作:
  1. 在 spec 的 BuildRequires 添加 simde-devel
  2. 修改源码: 用 #ifdef __loongarch__ 包裹 SIMDE include
  3. 纯 C 回退路径存在 → 直接指向回退分支
  4. 编译验证 → 功能测试 → 性能回归测试
```

#### 类别 D — 内联汇编

```
策略: 逐段替换汇编为 LA64 等价实现
工具: src/agent-docs/isa/ 下的 LA64 vs RV64/x86 指令映射表
操作:
  1. 列出所有内联汇编位置
  2. 分析每段的语义（查 isa/ YAML 文件）
  3. 查找 LA64 等价指令（查 isa/ARCH_DIFF.md）
  4. 编写 LA64 版本
  5. 用 #ifdef __loongarch__ 条件隔离
  6. 逐段编译验证
```

#### 类别 E — JIT/动态编译

```
策略: 理解 JIT 引擎的架构抽象层，实现 LA64 后端
关键依赖:
  - LLVM/Clang 对 LA64 的 preserve_none / preserve_all CC 支持
  - LA64 ELF 重定位类型 (B26, GOT_PC_HI20/LO12, PCALA_HI20/LO12)
  - LA64 调用约定（寄存器分配、栈帧布局）
操作:
  1. 检查 LLVM 版本及 LA64 后端完整性
  2. 补全缺失的 calling convention
  3. 适配重定位解析器
  4. 编写 LA64 指令发射逻辑
  5. 严格测试（单元测试 + 集成测试 + 压力测试）
```

---

## 5. 标准化移植流水线

### 5.1 流水线概述

```
┌────────────────────────────────────────────────────────────────────┐
│                       标准化移植流水线                               │
│                                                                    │
│  输入                       处理                         输出       │
│  ────                      ────                        ────        │
│                                                                    │
│  源码包                                                             │
│   │                                                                │
│   ▼                                                                │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐   │
│  │ 分类 │───→│ 收集 │───→│ 参考 │───→│ 适配 │───→│ 验证 │───→│ 交付 │   │
│  │ 判定 │    │ 信息 │    │ 构建 │    │ 构建 │    │ 测试 │    │      │   │
│  └─────┘    └─────┘    └─────┘    └─────┘    └─────┘    └─────┘   │
│     │          │          │          │          │          │       │
│     ▼          ▼          ▼          ▼          ▼          ▼       │
│  category   build-plan  x86.log    patches    test.md    RPM +     │
│  .json      .json       SRPM       .patch     report.json Patch     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 各阶段详细说明

#### 阶段 0: 分类判定

```bash
#!/bin/bash
# classify-package.sh — 自动分类脚本

ARCHIVE="$1"
CATEGORY="B"  # 默认: 便携 C/C++

# 解压到临时目录
TMPDIR=$(mktemp -d)
tar xf "$ARCHIVE" -C "$TMPDIR"

# 检查内联汇编
if grep -rqE '(asm\s*(__volatile__|volatile)?\s*\()|\.S"$' "$TMPDIR"; then
    CATEGORY="D"
# 检查 JIT
elif grep -rqE '(mmap.*PROT_EXEC|JIT|jit_compil|emit_code|stencil)' "$TMPDIR"; then
    CATEGORY="E"
# 检查 SIMD intrinsic
elif grep -rqE 'include.*(mm|avx|neon|sse)intrin\.h' "$TMPDIR"; then
    CATEGORY="C"
# 检查是否含编译型代码
elif ! grep -rqE '\.(c|cpp|cc|cxx|f|f90|f95|F|go|rs)$' "$TMPDIR"; then
    CATEGORY="A"
fi

rm -rf "$TMPDIR"
echo "{\"category\": \"$CATEGORY\", \"package\": \"$(basename $ARCHIVE)\"}"
```

#### 阶段 1: 信息收集

提取以下结构化信息：

```json
{
  "name": "example",
  "version": "1.2.3",
  "category": "B",
  "build_system": "autotools",           // autotools | cmake | meson | make | cargo | custom
  "has_spec": true,
  "dependencies": ["zlib-devel", "openssl-devel"],
  "x86_build_command": "rpmbuild -ba example.spec",
  "test_command": "make test",
  "arch_features": [],                    // [sse, avx, neon, asm] 等
  "notes": "使用 autotools，config.sub 可能过旧"
}
```

#### 阶段 2: X86-64 参考构建

在 x86_64 环境下执行，产出：

1. **构建日志** — `build-x86_64.log`
2. **SRPM 包** — `example-1.2.3-1.el9.src.rpm`
3. **测试结果** — `test-x86_64.log`
4. **构建信息 JSON** — 包含 CPU 型号、GCC 版本、所有依赖版本

```bash
# 参考构建命令模板
rpmdev-setuptree
cp example-1.2.3.tar.gz ~/rpmbuild/SOURCES/
cp example.spec ~/rpmbuild/SPECS/
sudo dnf builddep ~/rpmbuild/SPECS/example.spec
rpmbuild -ba ~/rpmbuild/SPECS/example.spec 2>&1 | tee build-x86_64.log
rpmbuild --rebuild ~/rpmbuild/SRPMS/example-*.src.rpm --with tests
```

#### 阶段 3: LoongArch64 适配

在 LA64 环境下执行：

```bash
# 1. 接收 x86_64 基线产物
# 2. 首次 mock 构建（预期可能失败）
mock --rebuild example-1.2.3-1.el9.src.rpm -r loongarch64 2>&1 | tee build-la64-attempt1.log

# 3. 分析错误 → 查阅 src/agent-docs/guidelines/la64-porting-guide.md
#    查找匹配的错误模式 → 应用修复

# 4. 修改 spec / 源码 / 补丁后重新构建
mock --rebuild example-1.2.3-2.el9.src.rpm -r loongarch64 2>&1 | tee build-la64-attempt2.log

# 5. 循环直到构建成功
```

**spec 适配标准操作**：

```spec
# 添加到现有 spec 文件的 LA64 适配块

# ---- 通用适配 (适用于大多数 B/C 类包) ----

# 0. 架构声明 (在 Preamble 段)
%ifarch loongarch64
ExclusiveArch:  loongarch64 x86_64 aarch64
%endif

# 1. 添加 LA64 特定依赖 (在 BuildRequires 段)
%ifarch loongarch64
BuildRequires:  simde-devel
BuildRequires:  autoconf automake libtool
%endif

# 2. 更新 autotools (在 %prep 段)
# (仅 autotools 项目需要)
%prep
%setup -q
%ifarch loongarch64
cp -f %{_datadir}/autoconf/build-aux/config.guess ./config.guess
cp -f %{_datadir}/autoconf/build-aux/config.sub   ./config.sub
%endif

# 3. 架构条件编译选项 (在 %build 段)
%build
%ifarch loongarch64
export CFLAGS="$CFLAGS -std=gnu89 -fcommon"
export CXXFLAGS="$CXXFLAGS -fpermissive"
%configure --disable-native-simd
%else
%configure
%endif
%make_build

# 4. 测试 (在 %check 段，可选)
%check
make test || %ifarch loongarch64 true %else exit 1 %endif
```

#### 阶段 4: 测试验证与交付

```bash
# 1. 安装测试
sudo dnf install ./example-1.2.3-2.loongarch64.rpm

# 2. 功能测试
example --version
example --self-test

# 3. 对比 x86_64 基线测试结果
diff <(grep PASS test-x86_64.log | sort) <(grep PASS test-la64.log | sort)

# 4. 交付物检查清单
# □ example-1.2.3-2.loongarch64.rpm   (二进制包)
# □ example-1.2.3-2.el9.src.rpm       (源码包)
# □ la64-adaptation.patch              (架构适配补丁)
# □ BUILD.md                            (构建记录)
# □ TEST-REPORT.md                      (测试对比报告)
```

---

## 6. 构建基础设施

### 6.1 rpmbuild 环境

```bash
# 安装基础工具
sudo dnf install -y rpm-build rpmdevtools rpmlint mock

# 创建 rpmbuild 目录树
rpmdev-setuptree
# 结果:
# ~/rpmbuild/
# ├── BUILD/       # 构建工作目录
# ├── BUILDROOT/   # 安装虚拟根目录
# ├── RPMS/        # 产出: 二进制 RPM
# │   └── loongarch64/
# ├── SOURCES/     # 源码 tarball 和补丁
# ├── SPECS/       # spec 文件
# └── SRPMS/       # 产出: 源码 RPM
```

### 6.2 mock 隔离构建

```bash
# mock 配置: /etc/mock/loongarch64.cfg
cat > /etc/mock/loongarch64.cfg << 'EOF'
config_opts['root'] = 'loongarch64'
config_opts['target_arch'] = 'loongarch64'
config_opts['legal_host_arches'] = ('loongarch64', 'x86_64')
config_opts['chroot_setup_cmd'] = 'install @buildsys-build'
config_opts['dist'] = 'el9'
config_opts['releasever'] = '9'
config_opts['dnf.conf'] = '''
[main]
keepcache=1
debuglevel=2
reposdir=/dev/null
logfile=/var/log/yum.log
retries=20
obsoletes=1
gpgcheck=0
assumeyes=1

[baseos]
name=Anolis OS $releasever - Base
baseurl=http://mirrors.openanolis.cn/anolis/$releasever/BaseOS/loongarch64/os/
enabled=1

[appstream]
name=Anolis OS $releasever - AppStream
baseurl=http://mirrors.openanolis.cn/anolis/$releasever/AppStream/loongarch64/os/
enabled=1

[powertools]
name=Anolis OS $releasever - PowerTools
baseurl=http://mirrors.openanolis.cn/anolis/$releasever/PowerTools/loongarch64/os/
enabled=1
'''
EOF

# 初始化 mock 环境
mock --init -r loongarch64

# 构建
mock --rebuild example-*.src.rpm -r loongarch64
```

### 6.3 工具链版本锁定

```yaml
# 推荐工具链版本（Anolis OS 9 / LoongArch64）
toolchain:
  gcc: "15.2.0"
  g++: "15.2.0"
  gfortran: "15.2.0"
  binutils: "2.42"
  glibc: "2.39"
  cmake: "3.28+"
  make: "4.4+"
  autoconf: "2.71+"
  automake: "1.16+"
  libtool: "2.4.7+"
  rpm-build: "4.18+"
  mock: "5.0+"

additional_tools:
  simde: "0.8.2+"        # SIMD 可移植层
  nasm: "2.16+"          # 仅 x86_64 场景
  qemu-user: "9.0+"      # 跨架构模拟（仅 x86_64 → LA64 场景）
```

---

## 7. Agent 辅助自动化

### 7.1 OpenClaw Agent 的角色

OpenClaw Agent 在移植流水线中承担以下角色：

| 阶段 | Agent 职责 | 自动/人工 |
|------|-----------|:---:|
| 分类判定 | 自动扫描源码特征，输出分类 | 🟢 自动 |
| 信息收集 | 读取 spec/ebuild/PKGBUILD/CMakeLists.txt | 🟢 自动 |
| 参考构建 | 在 x86_64 执行 rpmbuild，记录结果 | 🟢 自动 |
| 错误分析 | 匹配构建错误到已知模式库 | 🟡 半自动 |
| 补丁生成 | 根据错误模式生成修复补丁 | 🟡 半自动 |
| spec 更新 | 在 spec 中插入 LA64 适配块 | 🟢 自动 |
| 测试验证 | 运行测试套件，对比基线 | 🟢 自动 |
| 报告生成 | 生成 BUILD.md / TEST-REPORT.md | 🟢 自动 |
| 决策确认 | 复杂情况暂停询问用户 | 🔴 人工 |

### 7.2 Agent 知识源

```
OpenClaw Agent
    │
    ├─ src/agent-docs/README.md               ← 入口 + 行为契约
    ├─ src/agent-docs/business/domain-knowledge.md  ← 业务上下文
    ├─ src/agent-docs/guidelines/la64-porting-guide.md  ← 技术参考（核心）
    ├─ src/agent-docs/guidelines/decision-matrix.md     ← 决策树
    ├─ src/agent-docs/guidelines/common-pitfalls.md     ← 历史教训
    ├─ src/agent-docs/isa/ARCH_DIFF.md                  ← 指令集差异速查
    ├─ src/agent-docs/isa/loongarch64/*.yml             ← LA64 ISA 完整模型
    ├─ src/agent-docs/isa/riscv64/*.yml                 ← RV64 ISA 完整模型
    └─ docs/solutions/<pkg>/                            ← 历史移植方案
```

### 7.3 Agent 与流水线集成接口

```yaml
# OpenClaw 移植任务描述 (JSON)
task:
  package: "ffmpeg"
  version: "6.1.1"
  category: "C"               # Agent 判定或用户指定
  source: "ffmpeg-6.1.1.tar.xz"
  spec: "ffmpeg.spec"          # 可选: 上游 spec
  priority: "high"
  target_arch: "loongarch64"
  target_dist: "anolis9"

  # 基线信息 (阶段2完成后填入)
  baseline:
    x86_build_ok: true
    x86_test_pass_rate: "98% (245/250)"
    srpm_path: "ffmpeg-6.1.1-1.el9.src.rpm"

  # 移植结果 (阶段4完成后填入)
  porting:
    la64_build_ok: true
    la64_test_pass_rate: "96% (240/250)"
    patches_generated: 3
    spec_modified: true
    rpms_generated: ["ffmpeg-6.1.1-2.loongarch64.rpm", "ffmpeg-devel-6.1.1-2.loongarch64.rpm"]
```

---

## 8. 质量保证体系

### 8.1 质量门禁

每个阶段设置明确的通过标准：

```
阶段1 (信息收集)
  □ 构建系统已识别
  □ 依赖列表已提取
  □ 构建命令已验证

阶段2 (X86-64 参考构建)
  □ 构建成功 (exit code = 0)
  □ 测试套件执行完成
  □ 无严重编译警告
  □ SRPM 生成成功

阶段3 (LA64 移植构建)
  □ 构建成功 (exit code = 0)
  □ 无新增编译错误
  □ 架构适配修改已通过 git diff 记录

阶段4 (测试验证)
  □ LA64 测试通过率 ≥ 95% 基线通过率
  □ 无回归测试失败
  □ 性能不低于基线 80%（如适用）
```

### 8.2 测试策略

```
测试层级:
  L1 · 编译检查   → 确保 LA64 上能编译通过，无新增警告
  L2 · 单元测试   → make test / ctest / pytest
  L3 · 功能验证   → 手工运行关键功能（--version, 基本操作）
  L4 · 集成测试   → 与其他包的依赖关系验证
  L5 · 性能测试   → 与 x86_64 同场景性能对比（类别 C/D/E 必须）
```

### 8.3 回归防护

```bash
# 1. x86_64 回归检查（如果适配代码使用条件编译）
#    在 x86_64 上重新构建修改后的 SRPM
rpmbuild --rebuild example-modified.src.rpm

# 2. 跨架构行为一致性检查
#    对比 x86_64 和 LA64 的测试输出
diff test-x86_64-output.txt test-la64-output.txt

# 3. ABI 兼容性检查
abidiff libexample.so.x86_64 libexample.so.loongarch64
```

---

## 9. 移植数据库与知识沉淀

### 9.1 知识结构

每一次移植操作都产生结构化的知识条目：

```yaml
# 错误模式库条目
- id: "err-autotools-config-guess"
  pattern: "cannot guess build type.*loongarch"
  cause: "项目自带的 config.sub/guess 太旧"
  fix: "cp /usr/share/autoconf/build-aux/config.{sub,guess} ./"
  category: "B"
  frequency: "high"

- id: "err-asm-unsupported-arch"
  pattern: "unsupported architecture.*asm"
  cause: "内联汇编仅实现了 x86_64 或 aarch64 路径"
  fix: "添加 #ifdef __loongarch__ 纯 C 实现或 LA64 汇编"
  category: "D"
  frequency: "medium"

- id: "err-simd-x86-intrinsic"
  pattern: "fatal error: .*mmintrin.h.*No such file"
  cause: "X86 SSE/AVX intrinsic 在 LA64 上不可用"
  fix: "使用 #ifdef __loongarch__ 引入 SIMDE 头文件替代"
  category: "C"
  frequency: "high"
```

### 9.2 移植方案模板

每个成功移植的软件包产生一份标准方案文档，存放于 `docs/solutions/<pkg>/`：

```
docs/solutions/ffmpeg/
├── BUILD.md              # 完整构建步骤（双架构）
├── PATCHES.md            # 补丁说明（每处修改的原因和影响）
├── TEST-REPORT.md        # 测试对比报告
├── patches/              # 实际的 .patch 文件
│   ├── 0001-autotools.patch
│   ├── 0002-simd-simde.patch
│   └── 0003-la64-asm.patch
└── NOTES.md              # 杂项记录
```

### 9.3 统计与度量

```yaml
# 项目度量指标
metrics:
  total_packages_ported: 0          # 初版: 计数从0开始
  by_category:
    A: 0    # 架构无关
    B: 0    # 便携 C/C++
    C: 0    # SIMD/Intrinsic
    D: 0    # 内联汇编
    E: 0    # JIT

  avg_time_per_category:
    A: "unknown"
    B: "unknown"
    C: "unknown"
    D: "unknown"
    E: "unknown"

  build_success_rate: "N/A"
  test_pass_rate: "N/A"
  automated_rate: "N/A"  # 无需人工干预即可完成的百分比
```

---

## 10. 实施路线图

### Phase 1: 基础设施搭建（第 1-2 周）

- [ ] LA64 环境部署：安装 Anolis OS，配置 mock
- [ ] 工具链验证：GCC 15.2.0 + binutils 2.42 完整测试
- [ ] Agent 知识库加载：将 `src/agent-docs/` 作为 Agent 系统 prompt
- [ ] 流水线脚本开发：分类判定、信息收集、参考构建自动化脚本
- [ ] 第一个类别 A 包成功移植（验证流水线可用）

### Phase 2: 批量移植 — 低难度（第 3-4 周）

- [ ] 目标：完成 50 个 A/B 类软件包移植
- [ ] 积累错误模式库：每遇到一个新模式，写入 `docs/solutions/`
- [ ] Agent 自动化率验证：测量从接收到交付的人工介入次数
- [ ] 建立测试基线对比流程

### Phase 3: 中难度突破（第 5-8 周）

- [ ] 目标：完成 30 个 C 类包（ffmpeg, opencv, x264 等）
- [ ] SIMDE 集成最佳实践文档
- [ ] LA64 LSX/LASX intrinsic 编写指南
- [ ] 性能测试框架搭建

### Phase 4: 高难度攻克（第 9-16 周）

- [ ] 目标：完成 10 个 D 类包（openssl, glibc 关键模块等）
- [ ] 内联汇编移植工作流标准化
- [ ] D 类包的自动化程度评估（哪些子步骤可脚本化）
- [ ] JIT 类包预研（Python 3.14 JIT, V8 等）

### Phase 5: 生态闭环（持续）

- [ ] 将 LA64 RPM 提交到 Anolis OS 官方仓库
- [ ] 补丁向上游提交（upstream first 原则）
- [ ] 移植数据库公开，赋能社区
- [ ] CI/CD 集成：每次上游发版自动触发 LA64 构建

---

## 附录

### A. 相关文档索引

| 文档 | 路径 | 用途 |
|------|------|------|
| Agent 入口 | `src/agent-docs/README.md` | Agent 行为契约 |
| 业务领域知识 | `src/agent-docs/business/domain-knowledge.md` | 移植业务流程 |
| 架构概述 | `src/agent-docs/architecture/overview.md` | 双架构环境 + JIT |
| LA64 移植技术指南 | `src/agent-docs/guidelines/la64-porting-guide.md` | 核心技术参考 |
| 决策矩阵 | `src/agent-docs/guidelines/decision-matrix.md` | 不确定时决策 |
| 常见陷阱 | `src/agent-docs/guidelines/common-pitfalls.md` | 历史教训 |
| 安全操作规范 | `src/agent-docs/operations/safe-operations.md` | 操作分级 L0-L3 |
| ISA 模型中心 | `src/agent-docs/isa/README.md` | 指令集查阅指南 |
| ISA 差异速查 | `src/agent-docs/isa/ARCH_DIFF.md` | LA64 vs RV64 |

### B. 命令速查

```bash
# ── 环境确认 ──
uname -m                          # 确认架构
gcc -dumpmachine                  # 确认 GCC triplet
rpm --eval '%{_arch}'             # RPM 当前架构

# ── 构建工具 ──
rpmdev-setuptree                  # 初始化 rpmbuild 目录
rpmbuild -ba example.spec         # 构建二进制 + 源码包
rpmbuild -bs example.spec         # 仅构建源码包
mock --init -r loongarch64        # 初始化 mock chroot
mock --rebuild *.src.rpm -r loongarch64  # mock 构建

# ── 依赖管理 ──
sudo dnf builddep example.spec    # 安装构建依赖
dnf repoquery --requires example  # 查询运行时依赖
dnf repoquery --whatrequires example  # 反向依赖

# ── 验证 ──
rpmlint example.spec              # 检查 spec 规范
rpmlint example-*.rpm             # 检查 RPM 规范
rpm -qpl example-*.rpm            # 列出包内文件
rpm -qp --requires example-*.rpm  # 列出运行时依赖

# ── 包信息提取 ──
rpm -qp --qf '%{NAME} %{VERSION} %{ARCH}\n' *.rpm
grep -E '^(Name|Version|BuildRequires|ExclusiveArch):' example.spec
```

### C. 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| LA64 | LoongArch64 | 龙芯 64 位指令集架构 |
| LP64D | LP64 Data Model | 64 位 long/pointer + 硬件浮点 |
| LSX | Loongson SIMD eXtension | 128 位 SIMD 扩展 |
| LASX | Loongson Advanced SIMD eXtension | 256 位 SIMD 扩展 |
| LBT | Loongson Binary Translation | 二进制翻译辅助指令 |
| SIMDE | SIMD Everywhere | 可移植 SIMD intrinsic 封装库 |
| SRPM | Source RPM | 源码 RPM 包 |
| mock | — | chroot 隔离构建工具 |
| OpenClaw | — | 本项目的 AI Agent 名称 |

---

> **文档维护**
>
> | 日期 | 版本 | 更新内容 | 更新者 |
> |------|------|----------|--------|
> | 2026-06-08 | v0.1 | 初版，覆盖架构设计、5级分类体系、标准流水线、质量体系、路线图 | Hermes Agent |
