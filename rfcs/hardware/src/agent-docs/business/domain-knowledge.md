# 业务领域知识：LoongArch64 软件移植

> **Agent 阅读指引**
> 
> 本文档是理解OpenClaw移植业务的根基。在执行任何构建、编译或源码修改前，请确保你理解本节描述的移植上下文和流程。
> 
> 如果你对任何移植术语或流程有疑问，**不要猜测**，查阅此文档或询问用户。

## 1. 项目定位

### 1.1 OpenClaw 是什么

```yaml
name: OpenClaw
business_domain: "软件移植与跨架构适配"
primary_users: "龙芯(Loongson)生态开发者、Linux发行版维护者、开源社区"
core_mission: "将X86-64生态的开源软件移植到LoongArch64架构"
```

### 1.2 核心业务流程

```
接收/获取待移植项目源码
        │
        ▼
  ┌─────────────┐
  │ 阶段0: 源码保护 │  ← git init + initial commit（必须）
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ 阶段1: 信息收集 │  ← 查找构建文档、ebuild、PKGBUILD、CMakeLists等
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ 阶段2: X86-64  │  ← 在X86-64环境下建立参考构建基线
  │   参考构建    │    （验证项目本身可构建，记录步骤）
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ 阶段3: LA64   │  ← 在LoongArch64环境下执行构建
  │   移植验证   │    （识别架构差异问题，实施修复）
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ 阶段4: 移植   │  ← 生成补丁、撰写说明、提交交付
  │   提交       │
  └─────────────┘
```

**关键原则：没有X86-64基线，不开始LoongArch64移植。**

## 2. 关键业务实体

| 实体名称 | 含义 | Agent注意事项 |
|----------|------|---------------|
| X86-64 参考构建 | 在源架构上完成的首次成功构建 | 用于对比移植后的行为差异，必须记录完整步骤 |
| LoongArch64 (LA64) | 目标架构，龙芯自研指令集 | 注意与MIPS的区别，LA64是独立架构 |
| 初始提交 | `git init && git add -A && git commit -m "Initial import"` | 任何源码修改前的强制步骤 |
| 移植补丁 | 为适配LA64而做的源码修改集合 | 应可独立提取（git diff 或 format-patch） |
| 构建基线 | X86-64上成功构建的状态记录 | 包括依赖版本、配置参数、测试通过情况 |
| ebuild | Gentoo Linux的构建脚本 | 包含依赖、编译标志、安装规则等元信息 |
| PKGBUILD | Arch Linux/AUR的构建脚本 | 类似ebuild，但使用Arch的包管理格式 |
| spec | RPM系（RHEL/Fedora/CentOS）的构建脚本 | 包含BuildRequires、%build、%check、%files，用于rpmbuild打包 |

## 3. 移植规则与约束

### 3.1 不可违反的移植规则

以下规则具有业务强制性，任何Agent都必须遵守：

1. **源码备份优先**：修改任何源码前，必须确保Git已初始化且原始状态已提交
2. **基线先行**：必须先完成X86-64参考构建，确认项目在源架构上行为正常
3. **架构隔离**：X86-64构建产物和LoongArch64构建产物不可混用，交叉编译时明确指定工具链
4. **最小侵入**：移植修改应使用条件编译隔离架构差异，不破坏X86-64兼容性
5. **测试验证**：每个移植修复后，必须重新运行测试套件验证

### 3.2 业务术语表

| 术语 | 定义 | 技术映射 |
|------|------|----------|
| LA64 / LoongArch64 | 龙芯64位指令集架构 | `uname -m` 输出 `loongarch64`，GCC triplet: `loongarch64-linux-gnu` |
| X86-64 / AMD64 | 源架构，Intel/AMD 64位指令集 | `uname -m` 输出 `x86_64`，GCC triplet: `x86_64-linux-gnu` |
| Node方式 | 通过某种Node/容器环境切换架构 | 可能涉及Docker/QEMU/chroot等，需确认当前实际架构 |
| SSH方式 | 通过SSH连接到不同架构的远程主机 | 连接后执行 `uname -m` 确认远程架构 |
| 初始提交 | 首次将源码纳入Git版本控制 | `git init && git add -A && git commit -m "Initial import"` |
| 内联汇编 | 直接嵌入汇编代码的C/C++扩展 | X86的asm语句需替换为LA64版本或通用Intrinsic |
| 构建系统 | 管理编译过程的系统 | Makefile / CMake / Meson / autotools 等 |
| rpmbuild | RPM 包构建工具 | `rpmbuild -ba example.spec`，生成 .rpm 和 .src.rpm |
| mock | chroot 隔离构建工具 | `mock --rebuild example.src.rpm -r loongarch64-config` |
| make test | 常见的测试触发命令 | 也可能是 `ctest`, `ninja test`, `pytest`, `cargo test` |

## 4. 数据敏感性分级

| 等级 | 数据类型 | Agent操作限制 |
|------|----------|---------------|
| 🔴 极高 | 上游源码完整性 | **禁止在未备份的情况下修改或删除原始源码** |
| 🟠 高 | 构建配置、交叉编译工具链配置 | 修改前必须备份并确认 |
| 🟡 中 | 构建日志、测试输出、临时文件 | 可读取，清理时需谨慎 |
| 🟢 低 | 移植文档、分析笔记 | 可自由操作 |

## 5. 外部依赖与集成

```yaml
dependencies:
  - name: "GCC/Clang 工具链"
    criticality: "高"
    agent_note: "X86-64和LoongArch64需要各自对应的交叉编译器"
  
  - name: "构建依赖库"
    criticality: "高"
    agent_note: "ebuild/PKGBUILD中列出的依赖需在目标架构可用"
  
  - name: "QEMU 用户态模拟"
    criticality: "中"
    agent_note: "在X86主机上测试LoongArch64二进制时可能需要"
  
  - name: "Gentoo Portage / Arch Pacman / RPM dnf"
    criticality: "中"
    agent_note: "解析ebuild/PKGBUILD/spec获取依赖和构建标志"

  - name: "rpmdevtools / mock"
    criticality: "中"
    agent_note: "rpmdev-setuptree 创建构建目录，mock 在隔离 chroot 中构建 RPM"
```

## 6. 技术参考文档

遇到具体移植技术问题时，查阅：

| 问题类型 | 参考文档 |
|----------|----------|
| autotools/config.sub/guess 处理 | `guidelines/la64-porting-guide.md` 第1节 |
| SIMD/AVX/SSE/NEON 替换 | `guidelines/la64-porting-guide.md` 第2节 |
| `__rdtsc`, `_mm_pause`, `_mm_clflush` 等函数替换 | `guidelines/la64-porting-guide.md` 第3节 |
| LoongArch64 平台特性（地址模型、指令对齐、JIT） | `guidelines/la64-porting-guide.md` 第4节 |
| 构建失败快速索引 | `guidelines/la64-porting-guide.md` 第6节 |
| rpmbuild/spec 编写与 LA64 适配 | `guidelines/la64-porting-guide.md` 第9节 |

---

## 7. 构建信息收集指南

Agent在阶段1必须系统性地查找以下文件：

### 7.1 构建文档优先级

| 优先级 | 文件名/路径 | 内容 |
|--------|------------|------|
| 1 | `docs/build.md` | 项目专属构建指南 |
| 2 | `README.md` | 通常包含构建说明 |
| 3 | `BUILDING.md` | 专门的构建文档 |
| 4 | `INSTALL.md` | 安装说明，含构建步骤 |
| 5 | `*.ebuild` | Gentoo构建脚本（含依赖、use flag、编译选项） |
| 6 | `PKGBUILD` | Arch构建脚本（含依赖、makedepends、build函数） |
| 7 | `*.spec` | RPM构建脚本（含BuildRequires、%build、%check、%files） |
| 8 | `CMakeLists.txt` | CMake项目配置 |
| 9 | `configure.ac` / `configure` | Autotools项目 |
| 10 | `meson.build` | Meson项目 |
| 11 | `Makefile` | 直接Make项目 |

### 7.2 从ebuild提取的关键信息

```bash
# Agent应读取ebuild中的这些变量：
SRC_URI        # 源码下载地址
DEPEND         # 运行时依赖
RDEPEND        # 编译依赖
BDEPEND        # 构建时依赖
IUSE           # 可选功能标志
src_configure  # 配置步骤
src_compile    # 编译步骤
src_test       # 测试步骤
```

### 7.3 从PKGBUILD提取的关键信息

```bash
# Agent应读取PKGBUILD中的这些变量：
pkgname        # 包名
pkgver         # 版本
depends        # 运行时依赖
makedepends    # 编译依赖
source         # 源码地址
build()        # 构建函数
check()        # 测试函数
package()      # 打包函数
```

### 7.4 从spec提取的关键信息

```bash
# Agent应读取spec中的这些段和标签：
Name            # 包名
Version         # 版本
BuildRequires   # 编译依赖（最重要，决定LA64上是否可构建）
Requires        # 运行时依赖
ExclusiveArch   # 架构限制（检查是否已包含loongarch64）
Source0         # 源码下载地址
%build          # 编译步骤和选项
%check          # 测试命令
%install        # 安装规则（了解构建产物）
%ifarch         # 架构条件块（评估LA64适配插入点）
```

## 8. 业务上下文 FAQ（Agent 常见问题）

### Q: 用户只给了我源码，没有给构建文档怎么办？
A: 按优先级顺序查找构建系统文件。如果只有Makefile，直接阅读Makefile目标。如果是CMake项目，运行 `cmake -B build && cmake --build build`。如果完全无法识别构建系统，**暂停并询问用户**。

### Q: X86-64参考构建失败了，是否还要继续LoongArch64移植？
A: **不要**。先在X86-64上解决构建问题，区分这是项目本身的通用问题还是架构问题。如果确认是项目本身的Bug，记录后修复并重新构建。

### Q: 用户要求直接修改源码适配LA64，但没有要求X86-64基线怎么办？
A: **必须执行初始提交**。然后向用户说明基线原则，询问是否先进行X86-64参考构建。如果用户坚持跳过，记录此决策并继续，但需在报告中注明缺少基线对比。

### Q: 我通过SSH连接到了LoongArch64机器，如何确认？
A: 执行 `uname -m` 应输出 `loongarch64`；执行 `gcc -dumpmachine` 应输出 `loongarch64-linux-gnu`。

### Q: 构建产物（如.o文件、可执行文件）应该提交到Git吗？
A: **不应该**。确保 `.gitignore` 已包含build目录。如果项目没有.gitignore，创建一个。

## 9. 特殊项目类型：JIT/Copy-and-Patch 编译器移植

### 9.1 适用场景

Python 3.14 引入的 Copy-and-Patch JIT、JavaScript/WebAssembly 引擎（如 V8、SpiderMonkey）、LuaJIT、动态编译优化器等，涉及以下技术点：

| 领域 | 特殊要求 |
|------|---------|
| 运行时代码生成 | 需要 LA64 指令编码知识（固定4字节对齐） |
| ELF 重定位 | 需要了解 LA64 重定位类型（R_LARCH_B26, R_LARCH_GOT_PC_HI/LO12 等） |
| 调用约定 | `preserve_none`/`preserve_all` 等非标准 CC 可能在 LA64 缺失 |
| 内联汇编 | 需要替换 X86-64 特有的指令序列 |
| TCO/尾调用 | LA64 的尾调用使用 `b` 而非 `bl`（不修改 $ra） |

### 9.2 常见挑战

1. **LLVM 依赖陷阱**：JIT 通常依赖新版本 LLVM 提供的功能（如 `preserve_none`）。系统 LLVM 版本可能过旧，需从源码构建 LLVM 并为 LA64 后端打补丁。

2. **调用约定差异**：LLVM 的 `preserve_all`/`preserve_none` calling convention 在 X86-64 和 AArch64 有完整支持，但 LA64 是社区添加的较新后端，可能缺失。需在 `LoongArchCallingConv.td` 中定义 CSR，在 `LoongArchRegisterInfo.cpp` 中注册。

3. **重定位类型不匹配**：X86-64 使用 PC32、GOTPCREL 等重定位类型。LA64 使用 B26、GOT_PC_HI20/GOT_PC_LO12、PCALA_HI20/PCALA_LO12 等，需逐一适配。

4. **GCC/LLVM 调用约定不一致**：如果解释器由 GCC 编译、JIT 代码由 LLVM 生成，两者可能使用不同的默认调用约定。入口点函数需使用标准 CC 兼容两者。

5. **尾调用优化陷阱**：`return call(args)` 可能被编译器优化为 `b`（尾调用），跳过 stack frame 的 epilogue（callee-saved 寄存器恢复）。当 caller 是标准 CC、callee 是 `preserve_none` 时，这是个隐蔽的 bug。

### 9.3 典型修复流程

```bash
# 1. 检查 LLVM 版本
clang --version

# 2. 如果需要从源码构建 LLVM
# 参考 la64-porting-guide.md 第5节

# 3. 为 LA64 后端打 preserve_none 补丁
# 修改: LoongArchCallingConv.td, LoongArchRegisterInfo.cpp,
#       LoongArchISelLowering.cpp, Attr.td, LoongArch.h

# 4. 验证 preserve_none 生效
cat > /tmp/test_pn.c << 'EOF'
__attribute__((preserve_none)) int test_fn(int a, int b) {
    return a + b;
}
EOF
clang -O2 -S /tmp/test_pn.c
# 期望输出: ret 指令前无 addi.d $sp, $sp, -N 栈分配
```

---

> **维护记录**
> 
> | 日期 | 更新内容 | 更新者 |
> |------|----------|--------|
> | 2026-05-13 | 初始化模板 | Agent (Kimi) |
