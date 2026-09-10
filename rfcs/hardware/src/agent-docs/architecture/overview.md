# 架构概述：OpenClaw 移植环境

> **Agent 阅读指引**
> 
> 本文档描述OpenClaw移植工作的技术架构和环境配置。在进行构建、源码修改或跨架构操作时，请先理解本文档描述的双架构环境和工具链。

## 1. 双架构环境架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OpenClaw 双架构运行环境                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────┐           ┌─────────────────────┐            │
│   │   X86-64 构建环境    │           │  LoongArch64 构建环境 │            │
│   │  (参考构建/基线建立)  │           │   (移植验证/目标测试)  │            │
│   ├─────────────────────┤           ├─────────────────────┤            │
│   │  uname -m: x86_64   │           │ uname -m: loongarch64│            │
│   │  GCC: x86_64-linux- │           │ GCC: loongarch64-    │            │
│   │       gnu-gcc       │           │      linux-gnu-gcc   │            │
│   │  用途:              │           │ 用途:               │            │
│   │  - 阶段1: 信息收集   │           │ - 阶段3: 移植构建    │            │
│   │  - 阶段2: X86参考构建│           │ - 阶段3: 移植修复    │            │
│   │  - 验证项目通用性    │           │ - 阶段4: 移植测试    │            │
│   └─────────────────────┘           └─────────────────────┘            │
│            │                                  ▲                        │
│            │                                  │                        │
│            │         切换方式                  │                        │
│            │    ┌─────────────────┐           │                        │
│            └───▶│  Node / SSH     │───────────┘                        │
│                 │  / 容器 / QEMU  │                                    │
│                 └─────────────────┘                                    │
│                                                                         │
│   共同要求:                                                             │
│   - 源码仓库: (用户指定，由用户在加载项目时提供路径)                    │
│   - Git版本控制: 必须初始化并提交                                        │
│   - 构建文档: 记录于项目内或本Agent文档中                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. 目录结构约定

```
(用户指定的项目目录)/                 # 项目根目录（工作目录）
├── src/                               # 业务源代码（Agent主要操作区域）
│   ├── agent-docs/                    # Agent行为指南（本文档）
│   │   ├── README.md
│   │   ├── business/
│   │   ├── architecture/
│   │   ├── operations/
│   │   └── guidelines/
│   └── [待移植项目源码]/                # 用户提供的待移植项目
│
├── [build/]                           # 构建目录（建议统一命名）
├── [docs/]                            # 项目文档（优先查找build.md）
├── [*.ebuild]                         # Gentoo构建脚本（如有）
├── [PKGBUILD]                         # Arch构建脚本（如有）
└── .git/                              # Git版本控制（必须存在）
```

### 2.1 Agent 目录操作规范

| 目录/文件 | 用途 | Agent权限 | 注意事项 |
|-----------|------|-----------|----------|
| `src/agent-docs/` | Agent文档 | **读写** | 修改后同步更新 |
| `src/[项目源码]/` | 待移植项目 | 读写 | 修改前必须Git提交 |
| `build/` | 构建产物目录 | 读写 | 不提交到Git |
| `*.ebuild` | Gentoo脚本 | 只读 | 提取信息后引用 |
| `PKGBUILD` | Arch脚本 | 只读 | 提取信息后引用 |
| `.git/` | 版本控制 | **只读** | 禁止手动修改 |

## 3. 核心模块说明

### 3.1 移植阶段模块

| 模块名 | 路径/职责 | 关键操作 | 执行架构 |
|--------|-----------|----------|----------|
| 源码保护 | Git初始化与提交 | `git init && git add -A && git commit -m "Initial import"` | 当前架构 |
| 信息收集 | 扫描构建文档和脚本 | 读取README/ebuild/PKGBUILD/CMakeLists.txt | 当前架构 |
| X86-64参考构建 | 建立构建基线 | 执行文档记录的构建步骤，保存日志 | X86-64 |
| LoongArch64移植构建 | 在目标架构执行构建 | 重复X86步骤，记录差异 | LoongArch64 |
| 移植修复 | 修改源码适配LA64 | 条件编译、内联汇编替换、类型调整 | LoongArch64 |
| 测试验证 | 运行测试套件 | `make test` / `ctest` / 项目自定义测试 | LoongArch64 |
| 补丁生成 | 导出移植修改 | `git diff` / `git format-patch` | 当前架构 |

### 3.2 数据流

```
原始源码（上游tarball/git clone）
    │
    ▼
Git初始化 + Initial Commit  ←──── 强制步骤，不可逆
    │
    ▼
构建信息提取 ─────┬─────▶ ebuild依赖分析
    │           ├─────▶ PKGBUILD依赖分析
    │           └─────▶ 构建系统识别
    │
    ▼
X86-64参考构建 ────┬─────▶ 成功：记录基线
    │             └─────▶ 失败：修复通用问题
    │
    ▼
架构切换（Node/SSH/QEMU）
    │
    ▼
LoongArch64构建 ───┬─────▶ 成功：记录移植步骤
    │             └─────▶ 失败：分析架构差异 → 移植修复
    │
    ▼
测试验证 ──────────┬─────▶ 通过：移植完成
    │             └─────▶ 失败：继续修复
    │
    ▼
补丁导出 + 移植报告
```

## 4. 技术栈

```yaml
# 移植工作技术栈
source_architecture:
  name: "X86-64"
  uname_m: "x86_64"
  gcc_triplet: "x86_64-linux-gnu"
  typical_distro: "[待确认 - Gentoo/Arch/Debian等]"

target_architecture:
  name: "LoongArch64"
  uname_m: "loongarch64"
  gcc_triplet: "loongarch64-linux-gnu"
  typical_distro: "[待确认 - Loongnix/Arch Linux等]"

environment_switch_methods:
  - method: "Node方式"
    description: "通过Node环境切换（具体机制需确认）"
    check_command: "uname -m"
  - method: "SSH方式"
    description: "SSH连接到LoongArch64远程主机"
    check_command: "ssh [host] 'uname -m'"
  - method: "容器/QEMU"
    description: "使用Docker或QEMU用户态模拟"
    check_command: "docker exec [container] uname -m"

build_systems_supported:
  - "Makefile (GNU Make)"
  - "CMake"
  - "Meson"
  - "Autotools (autoconf/automake)"
  - "Cargo (Rust)"
  - "其他（需在信息收集阶段识别）"

package_scripts:
  - "Gentoo ebuild (*.ebuild)"
  - "Arch PKGBUILD (PKGBUILD)"
  - "Debian rules (debian/rules)"
  - "RPM spec (*.spec)"
```

## 5. 环境识别命令速查

Agent在切换环境后，必须执行以下命令确认当前状态：

```bash
# 1. 确认CPU架构
uname -m
# 预期输出: x86_64 (X86环境) 或 loongarch64 (LA64环境)

# 2. 确认GCC目标平台
gcc -dumpmachine
# 预期输出: x86_64-linux-gnu 或 loongarch64-linux-gnu

# 3. 确认GCC版本（某些项目对GCC版本敏感）
gcc --version

# 4. 确认当前工作目录和Git状态
pwd && git status

# 5. 如果是CMake项目，确认CMake版本
cmake --version

# 6. 记录环境到移植日志
echo "=== Environment Check ===" > /tmp/port-env.log
uname -a >> /tmp/port-env.log
gcc -dumpmachine >> /tmp/port-env.log
gcc --version >> /tmp/port-env.log
```

## 6. 测试策略

```yaml
test_levels:
  x86_64_baseline:
    description: "在X86-64上运行完整测试，建立预期行为基线"
    obligation: "移植前必须完成"
    commands:
      - "make test"
      - "ctest --output-on-failure"
      - "项目自定义测试命令"
  
  loongarch64_port:
    description: "在LoongArch64上运行相同测试，对比结果"
    obligation: "每个移植修复后必须运行"
    commands:
      - "与X86-64相同的测试命令"
  
  regression_check:
    description: "验证移植未破坏X86-64兼容性（如适用）"
    obligation: "最终提交前建议执行"
    note: "如果修改使用了条件编译，应在X86-64上重新构建验证"

agent_obligation: >
  Agent必须在移植报告中记录：
  1. X86-64测试通过率
  2. LoongArch64移植前测试通过率（预期为0或很低）
  3. LoongArch64移植后测试通过率
  4. 未通过的测试及其原因分析
```

## 7. 跨架构 JIT 编译器移植关键点

### 7.0 LA64 与 AArch64 指令语义差异（JIT 相关）

**这是从 AArch64 移植 JIT 编译器到 LA64 时最常见的错误来源。**

| 特性 | AArch64 | LA64 | 差异影响 |
|------|---------|------|---------|
| PC译址 | `ADRP` = `(PC & ~0xFFF) + (imm<<12)` | `PCADDU12I` = `PC + (imm<<12)` | LA64 不做页对齐，`LD.D` 偏移必须为 `target−PC−(imm<<12)` |
| 条件跳转 | `B.cond` offset26 (±128MB) | `BEQZ`/`BNEZ` offset21 | LA64 的条件分支范围是 ±1MB（±2^21），无条件 `B` 是 ±128MB |
| 指令对齐 | 所有指令4字节 | 所有指令4字节 | 相同 |
| 加载对齐 | `LDR` 可非对齐（慢） | `LD.D` 必须 8 字节对齐 | JIT 跳板中的地址数据必须 8 对齐 |
| 左移加法 | `ADD Xd, Xn, Xm, lsl #n` | `ALSL.D rd, rj, rk, sa` | LA64 的 `ALSL.D` 是 `rd = rk + (rj << sa)`，sa 是直接移位量 |
| 尾调用 | `B` 不修改 LR | `B` 不修改 $ra | 相同 |
| 函数返回 | `RET` = `BR LR` | `RET` = `JR $ra` | 相同 |

### 7.1 Copy-and-Patch JIT 数据流（Python 3.14）

```
Python Bytecode (tier1)
    │
    │  Tier2 Optimizer (uop trace)
    ▼
Optimized UOp Trace ─────→ Tier2 Bytecode Executor（有 LA64 bug）
    │                              ║
    │  _PyJIT_Compile()            ║ 仅非 JIT 编译的 trace 走此路径
    ▼                              ║
JIT Stencils (mmap'd code)        ║
    │                              ║
    ├─ [Shim: standard CC] ────────╫── 被 GCC 解释器调用
    ├─ [Stencil 0..N: preserve_none]║  ⟵ musttail 链
    └─ [_FATAL_ERROR]              ║
              │                    ║
              ▼                    ▼
       返回至解释器或崩溃        Tier2 crash
```

### 7.2 关键依赖链

| 组件 | 技术栈 | LA64 支持情况 |
|------|--------|-------------|
| LLVM | clang-19 + preserve_nonecc 补丁 | ✅ 已移植，CSR_NoneRegs |
| JIT Build | `Tools/jit/build.py` + clang → stencil .h | ✅ 已移植 |
| Runtime Patcher | `Python/jit.c` 重定位 + patch | ✅ 已移植 |
| Shim 入口 | `Tools/jit/shim.c` | ✅ 标准 CC + volatile 防尾调用 |
| Tier2 Executor | `Python/executor_cases.c.h` | ❌ 有 LA64 bug（~5000次崩溃） |

### 7.3 构建命令速查

```bash
# 1. 使用 clang-19 生成 JIT stencils（每次 LLVM 更新后需重新生成）
cd ~/src/Python-3.14.5
CLANG=clang-19 python3 Tools/jit/build.py loongarch64-unknown-linux-gnu \
  --output-dir . --pyconfig-dir . --force

# 2. 构建 Python 本体
export PATH="$HOME/src/llvm-project-19.1.7.src/build/bin:$PATH"
make -j$(nproc)

# 3. 测试（注意 tier2 bug）
PYTHON_JIT=yes ./python -c "for i in range(4000): x = i"  # OK
PYTHON_JIT=0 ./python -c "for i in range(100000): x = i"  # OK（完全禁用 JIT+tier2）
```

### 7.4 环境变量参考

| 变量 | 含义 | 行为 |
|------|------|------|
| `PYTHON_JIT=0` | 完全禁用 JIT 和 tier2 | 不使用 stencil，基础解释器执行 |
| `PYTHON_JIT=yes` | 启用 JIT 编译 | 热循环编译为 stencil 代码，short traces 走 tier2 |
| `PYTHON_JIT=no` | 仅 tier2 优化，不编译 JIT | 热循环仅 tier2 优化执行，不做 JIT 编译 |

---

## 8. 部署与发布

### 8.1 环境定义

| 环境 | 架构 | 用途 | 数据性质 | Agent操作限制 |
|------|------|------|----------|---------------|
| X86-64本地 | x86_64 | 阶段1-2：信息收集与参考构建 | 源码 | 常规操作 |
| X86-64容器 | x86_64 | 隔离的参考构建环境 | 源码+构建产物 | 常规操作 |
| LA64 Node | loongarch64 | 阶段3：移植构建与验证 | 源码+构建产物 | 需谨慎修改源码 |
| LA64 SSH | loongarch64 | 远程移植验证 | 源码+构建产物 | 需谨慎修改源码 |
| LA64 QEMU | loongarch64 | 用户态模拟测试 | 源码+构建产物 | 性能受限，仅测试 |

### 8.2 移植交付物

```yaml
deliverables:
  - item: "Git diff / Patch 文件"
    format: "*.patch 或 git format-patch 输出"
    content: "所有为适配LA64的源码修改"
  
  - item: "移植说明文档"
    format: "Markdown 或纯文本"
    content: "修改原因、影响范围、测试结论"
  
  - item: "构建步骤记录"
    format: "Shell脚本或Markdown"
    content: "X86-64和LA64的完整构建命令"
  
  - item: "测试报告"
    format: "Markdown"
    content: "双架构测试对比结果"
```

---

> **维护记录**
> 
> | 日期 | 更新内容 | 更新者 |
> |------|----------|--------|
> | 2026-05-13 | 初始化并填入LoongArch64移植架构信息 | Agent (Kimi) |
