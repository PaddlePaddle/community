# LoongArch64 移植技术指南

> **Agent 阅读指引**
>
> 本文档是 OpenClaw 移植工作的**核心技术参考**。当你在处理 LoongArch64 移植时遇到具体的技术问题（构建失败、编译错误、架构特性不匹配等），应首先查阅本文档。
>
> 本文档按问题类型组织，Agent 可根据错误信息快速定位到对应章节。

---

## 1. Autoconf/Automake 项目处理

### 1.1 识别方式

项目使用 autotools 的典型标志：
```
存在 config.sub
存在 config.guess
存在 configure.ac 或 configure.in
存在 Makefile.am
```

### 1.2 标准处理流程

当确认项目使用 autotools 后，按以下顺序执行：

**步骤 1：更新 config 脚本**

系统级的 `config.sub` 和 `config.guess` 通常已经支持 `loongarch64`，将其覆盖到项目中：

```bash
# 查找系统 config 脚本
find /usr -name "config.sub" 2>/dev/null
find /usr -name "config.guess" 2>/dev/null

# 典型路径（根据发行版可能不同）：
# /usr/share/automake-*/config.sub
# /usr/share/libtool/build-aux/config.sub
# /usr/share/autoconf/build-aux/config.sub

# 覆盖到项目路径
cp /usr/share/automake-1.16/config.sub ./config.sub
cp /usr/share/automake-1.16/config.guess ./config.guess
```

> **Agent 注意**：不要直接使用 `cp` 硬编码路径。先用 `find /usr -name "config.sub"` 查找可用副本，选择版本最新的那个。

**步骤 2：重新生成 configure**

根据项目情况选择以下方式之一：

```bash
# 方式 A：项目有 bootstrap 脚本
if [ -f bootstrap ]; then
    ./bootstrap
elif [ -f autogen.sh ]; then
    ./autogen.sh

# 方式 B：项目没有 bootstrap，但有 configure.ac/configure.in
elif [ -f configure.ac ] || [ -f configure.in ]; then
    autoreconf -fvis

# 方式 C：项目已有 configure 脚本（可能是预生成的）
else
    # 如果 configure 已存在且看起来可用，可直接使用
    # 但建议优先重新生成，以确保 config.sub/guess 的更新生效
    echo "已有 configure，建议重新生成以确保 loongarch64 支持"
fi
```

**步骤 3：执行 configure**

```bash
./configure
# 或根据项目文档指定参数
./configure --prefix=/usr
```

**步骤 4：构建**

```bash
make -j$(nproc)
```

---

## 2. 架构特性与 SIMD 处理

### 2.1 识别项目使用的架构特性

Agent 需要扫描源码，识别项目是否使用了架构特定的 SIMD 指令集：

**X86 特性识别**：
```bash
# 搜索 X86 SIMD 头文件
grep -rE "include.*[instpex]?mmintrin\.h" src/ || true
grep -rE "include.*xsaveintrin\.h" src/ || true
grep -rE "include.*avx.*intrin\.h" src/ || true

# 常见 X86 SIMD 头文件列表：
# - immintrin.h      (AVX/AVX2/AVX-512)
# - emmintrin.h      (SSE2)
# - xmmintrin.h      (SSE)
# - pmmintrin.h      (SSE3)
# - tmmintrin.h      (SSSE3)
# - smmintrin.h      (SSE4.1)
# - nmmintrin.h      (SSE4.2)
# - ammintrin.h      (SSE4A)
# - mmintrin.h       (MMX)
```

**ARM 特性识别**：
```bash
# 搜索 ARM SIMD 头文件
grep -rE "include.*arm_neon\.h" src/ || true
grep -rE "include.*arm_sve\.h" src/ || true
```

### 2.2 处理策略决策树

```
发现架构特性代码
  │
  ▼
┌─────────────────────────────┐
│ 该特性块是否有纯C回退实现？   │
│ 或 #else 分支为空（不优化）？ │
└─────────────────────────────┘
  │
  ├─ 是 ──▶ 添加 #elif defined(__loongarch_lp64) 条件
  │          指向纯C实现或不优化分支
  │
  └─ 否 ──▶ 使用 SIMDE 方案（见下文）
```

### 2.3 SIMDE 替换方案

当项目使用 X86 的 AVX/SSE/MMX 或 ARM 的 NEON/SVE，且没有纯 C 回退时，使用 **SIMDE**（SIMD Everywhere）项目提供的可移植头文件。

**操作步骤**：

```c
// 原始代码（X86）：
#include <immintrin.h>

// 移植后代码：
#if defined(__loongarch_lp64)
#  include <simde/x86/avx2.h>    // 或对应SIMDE头文件
#elif defined(__x86_64__)
#  include <immintrin.h>
#elif defined(__aarch64__)
#  include <arm_neon.h>
#else
#  error "Unsupported architecture"
#endif
```

**SIMDE 头文件映射参考**：

| 原始 X86 头文件 | SIMDE 替换头文件 |
|----------------|-----------------|
| `<immintrin.h>` | `<simde/x86/avx2.h>` 或 `<simde/x86/avx512.h>` |
| `<emmintrin.h>` | `<simde/x86/sse2.h>` |
| `<xmmintrin.h>` | `<simde/x86/sse.h>` |
| `<pmmintrin.h>` | `<simde/x86/sse3.h>` |
| `<tmmintrin.h>` | `<simde/x86/ssse3.h>` |
| `<smmintrin.h>` | `<simde/x86/sse4.1.h>` |
| `<nmmintrin.h>` | `<simde/x86/sse4.2.h>` |
| `<mmintrin.h>` | `<simde/x86/mmx.h>` |

| 原始 ARM 头文件 | SIMDE 替换头文件 |
|----------------|-----------------|
| `<arm_neon.h>` | `<simde/arm/neon.h>` |
| `<arm_sve.h>` | SIMDE 支持有限，需评估 |

> **Agent 注意**：SIMDE 可能需要作为依赖安装。检查发行版是否提供 `simde` 包，或需要从源码引入。

---

## 3. 特定函数替换表

以下函数在 LoongArch64 上需要替换或实现：

### 3.1 时间戳读取：`__rdtsc()`

**X86 原始代码**：
```c
uint64_t val = __rdtsc();
```

**LoongArch64 替换**：
```c
#if defined(__loongarch_lp64)
    uint64_t val;
    asm ("rdtime.d %0" : "=r"(val));
#else
    uint64_t val = __rdtsc();
#endif
```

### 3.2 自旋锁暂停：`_mm_pause()`

**X86 原始代码**：
```c
_mm_pause();  // SSE2 指令，提示CPU当前在自旋等待
```

**LoongArch64 替换**：
```c
#if defined(__loongarch_lp64)
    // LoongArch64 没有提供自旋锁等待指令，替换为空
#else
    _mm_pause();
#endif
```

> **解释**：LoongArch64 架构没有与 `PAUSE` 等价的指令，直接省略即可。

### 3.3 内存屏障：`_mm_clflush` / `_mm_sfence` / `_mm_mfence`

**X86 原始代码**：
```c
_mm_clflush(ptr);
_mm_sfence();
_mm_mfence();
```

**LoongArch64 替换**：
```c
#if defined(__loongarch_lp64)
    // dbar 0 是 LoongArch64 的内存屏障指令
    asm volatile ("dbar 0" ::: "memory");
#else
    _mm_clflush(ptr);
    _mm_sfence();
    _mm_mfence();
#endif
```

> **Agent 注意**：`dbar 0` 是 LoongArch64 的通用内存屏障。`_mm_clflush` 的缓存行刷新语义在 LA64 上没有直接等价物，如果项目确实需要缓存控制（如JIT编译器），需要更深入的分析。对于一般移植，`dbar 0` 作为内存屏障通常足够。

### 3.4 栈指针读取：`mv %0, sp`

**问题**：部分项目使用内联汇编将栈指针复制到通用寄存器。

**X86/MIPS 原始代码（示意）**：
```c
// X86: mov %rax, %rsp
// MIPS: move %0, $sp
// 某些项目可能写为：mv %0, sp
```

**LoongArch64 替换**：
```c
#if defined(__loongarch_lp64)
    // LoongArch64 中，mv 指令不存在，使用 addi.d 实现相同语义
    register void* sp_reg;
    asm ("addi.d %0, $sp, 0" : "=r"(sp_reg));
#else
    // 原架构实现
#endif
```

> **解释**：LoongArch64 没有 `mv`（move）指令。`addi.d %0, $sp, 0` 将 `$sp` + 0 的结果存入目标寄存器，等价于复制栈指针值。

---

## 4. LoongArch64 平台特性

### 4.1 GLIBC 绝对地址

```
在 LoongArch64-linux-gnu 下：
- glibc 使用绝对地址（与 x86-64 行为相同）
- 与 mips64、riscv64 的结构不同
```

**对移植的影响**：
- 如果项目有针对 MIPS64/RISC-V64 的特殊处理（如重定位、地址计算），不要将其直接套用到 LoongArch64
- LoongArch64 的地址模型更接近 X86-64

### 4.2 CMake SYSTEM_PROCESSOR

在 CMake 项目中，架构检测可能使用 `CMAKE_SYSTEM_PROCESSOR`：

```cmake
# 原始代码（可能只处理了 x86_64 和 aarch64）：
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH_AARCH64 TRUE)
endif()

# 移植后代码：
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_AARCH64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|loong64")
    set(ARCH_LOONGARCH64 TRUE)
endif()
```

> **重要**：`CMAKE_SYSTEM_PROCESSOR` 在 LoongArch64 上的值可能是 `"loongarch64"` 或 `"loong64"`，两种情况都要处理。

### 4.3 指令对齐与 JIT/重定位

**固定 4 字节指令对齐**：
- LoongArch64 的指令长度固定为 4 字节（32位），没有变长指令
- 这与 X86-64 的变长指令（1-15字节）完全不同
- 与 RISC-V64 类似（也是固定 4 字节）

**JIT 编译器移植建议**：
- LoongArch64 与 RISC-V64 的基础指令相似度较高
- 轻量级 JIT 可以参考：
  1. RISC-V64 的实现
  2. GCC 源码中的 LoongArch64 后端
  3. LLVM 源码中的 LoongArch64 后端
  4. LoongArch64 基础指令编码文档（官方手册）

**重定位处理**：
- 由于指令对齐固定，重定位类型和计算方式与 X86-64 不同
- 参考 ELF 规范中 LoongArch64 的重定位类型定义
- 使用 LA64 工具链提供的重定位宏（如 `_LOONGARCH64_` 相关定义）

### 4.4 关键指令差异：PCADDU12I vs AArch64 ADRP

**这是 JIT 移植中最容易踩的坑。**

AArch64 的 `ADRP` 指令行为：
```asm
// AArch64 ADRP: Xt = (PC & ~0xFFF) + (imm21 << 12)
// 先对 PC 做页对齐（清除低12位），再加偏移
adrp x0, page_of(symbol)
add  x0, x0, #offset_of(symbol)   // 或 ldr x0, [x0, #offset]
```

LA64 的 `PCADDU12I` 指令行为：
```asm
// LA64 PCADDU12I: rd = PC + (imm20 << 12)
// PC 不做页对齐！PC 的低12位保留
pcaddu12i $a0, page_diff
addi.d    $a0, $a0, offset   // 或 ld.d $a0, $a0, offset
```

**差异影响**：AArch64 的 `ADRP` 清除 PC 的低12位，然后 `LDR`/`ADD` 使用 `symbol & 0xFFF` 作为偏移。而 LA64 的 `PCADDU12I` 保留 PC 的低12位，`LD.D`/`ADDI.D` 的偏移必须是 `target − (PC + imm20<<12)`，**不能直接用 `target & 0xFFF`**。

**正确实现（以 GOT 加载为例）**：
```c
// 正确做法：diff = target_address − instruction_address
// 然后拆分：imm20 = diff >> 12, si12 = diff & 0xFFF（需处理12位符号扩展）
void patch_loongarch64_20r(unsigned char *location, uint64_t value) {
    int64_t diff = (int64_t)value - (int64_t)(uintptr_t)location;
    int64_t imm20_val = diff >> 12;
    int64_t low12 = diff & 0xFFF;
    if (low12 > 2047) imm20_val += 1;  // 补偿12位符号扩展
    set_bits(loc32, 5, imm20_val & 0xFFFFF, 0, 20);
}

void patch_loongarch64_12(unsigned char *location, uint64_t value) {
    int64_t diff = (int64_t)value - (int64_t)(uintptr_t)location;
    int64_t si12_val = diff & 0xFFF;
    if (si12_val > 2047) si12_val -= 4096;  // 12位有符号范围-2048..+2047
    set_bits(loc32, 10, si12_val & 0xFFF, 0, 12);
}
```

> **错误做法**（拷贝 AArch64 实现）：`imm20 = (target_page − pc_page) >> 12`、`imm12 = target & 0xFFF`。这会加载错误的地址，导致 JIT 代码崩溃。

### 4.5 ALSL.D 指令语义

LA64 的 `ALSL.D rd, rj, rk, sa`（左移加法双字）指令：
```asm
// ALSL.D: rd = rk + (rj << sa)
// 注意：sa 是直接移位量，不是 (sa+1)！
alsl.d $a0, $a1, $a0, 1   # $a0 = $a0 + ($a1 << 1) = funcobj + (target * 2)
```

**常见用途**：
- 计算字节码地址：`code_object + (target << 1) + bytecode_offset`
  - 其中 `target << 1` = `target * sizeof(_Py_CODEUNIT)` = 代码单元转字节
  - `bytecode_offset` = 从 code_object 到 co_code_adaptive 的偏移

> **注意**：`ALSL.D` 的 `sa` 是**直接移位量**（与 ARM 的 `LSL` 等不同），不是 `sa+1`。验证方式：`alsl.d $a0, $a1, $a2, 1` 计算 `$a0 = $a2 + ($a1 << 1)`。

### 4.6 内存对齐要求

LA64 严格要求对齐访问：

| 指令 | 对齐要求 | 非对齐后果 |
|------|---------|-----------|
| `LD.D` / `ST.D` | 8字节 | SIGBUS |
| `LD.W` / `ST.W` | 4字节 | SIGBUS |
| `LD.H` / `ST.H` | 2字节 | SIGBUS |
| `LDX.D` / `STX.D` | 8字节 | SIGBUS |
| `FST.D` / `FLD.D` | 8字节 | SIGBUS |

**JIT 移植特别注意**：
- 跳板（trampoline）中的地址数据必须 8 字节对齐
- `pcaddu12i` + `ld.d` 中的 `ld.d` 偏移必须使目标地址 8 对齐
- 如果跳板大小为 20 字节，偏移 12 处不是 8 对齐 → 需改为 24 字节跳板，地址放在偏移 16

---

## 5. LLVM 相关注意事项

### 5.1 LLVM 版本兼容性

在 LA64 平台（Anolis OS 23）上，系统默认 LLVM 版本通常落后上游（如 LLVM 18 vs 上游 LLVM 19），需要注意：

| 依赖 | 系统版本 | 目标版本 | 处理方式 |
|------|---------|---------|---------|
| LLVM/clang | 18.1.8 | 19.1.7+ | 从源码编译，需 LA64 GCC 作为宿主编译器 |
| preserve_none | llvm18 不支持 | LLVM 19+ 需补丁 | LLVM 19 官方 La64 后端未含 preserve_nonecc，需手动补丁 |

### 5.2 LoongArch64 preserve_none 调用规范补丁

Python 3.14 的 Copy-and-Patch JIT 使用 `__attribute__((preserve_none))` 消除 stencil 函数的栈帧。LLVM 19 在 X86-64/AArch64 后端支持此属性，但 LA64 后端没有。

**需修改的文件（LLVM 19.1.7）：**

1. **`clang/lib/Basic/Targets/LoongArch.h`** — 声明 `setSupportedOpenCLOptimization()` 防止 ICE（clang-18 作为宿主编译器时）
2. **`clang/include/clang/Basic/Attr.td`** — 为 LoongArch 启用 `preserve_none` 属性
3. **`llvm/lib/Target/LoongArch/LoongArchCallingConv.td`** — 定义 `CSR_NoneRegs` = (R1, R22) 作为 preserve_none 的寄存器集
4. **`llvm/lib/Target/LoongArch/LoongArchRegisterInfo.cpp`** — 在 `getCalleeSavedRegs()` 和 `getCallPreservedMask()` 中为 `PreserveNone` 返回 `CSR_NoneRegs`
5. **`llvm/lib/Target/LoongArch/LoongArchISelLowering.cpp`** — 添加 `PreserveNone` 的 LowerFormalArguments 处理

**`CSR_NoneRegs` 的选择理由：**
- 仅保存 R1 ($ra) 和 R22 ($fp/s0)
- 所有 $s0-$s8 等标准 callee-saved 寄存器不再保存
- 浮点寄存器也不保存（不涉及浮点操作的 JIT stencil）
- stencil 的 musttail 链中使用 `b`（branch, 不修改 $ra），$ra 稳定贯穿全链

**参考 calling convention 注册集定义：**
```tablegen
// Standard LP64 callee-saved registers
def CSR_ILP32S_LP64S
    : CalleeSavedRegs<(add R1, (sequence "R%u", 22, 31))>;

// preserve_none: only save $ra and $fp
def CSR_NoneRegs : CalleeSavedRegs<(add R1, R22)>;

// For GHC calling convention (truly no preserved regs)
def CSR_NoRegs : CalleeSavedRegs<(add)>;
```

### 5.3 LLVM 19 构建注意事项（宿主编译）

LA64 上编译 LLVM 耗时较长，建议：
```bash
# 使用 clang-18 作为宿主编译器（GCC 12 可能触发 ICE）
export CC=clang-18 CXX=clang++-18

# 禁用不必要的组件加速编译
cmake -GNinja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="LoongArch;X86" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1  # LA64 链接器内存受限
```

---

## 6. 快速问题索引

| 错误/问题关键词 | 对应章节 | 处理方案 |
|----------------|---------|---------|
| `config.sub: loongarch64-unknown-linux-gnu: not supported` | 第1节 | 覆盖系统 config.sub/guess，重新 autoreconf |
| `include <immintrin.h>` 编译失败 | 第2节 | 使用 SIMDE 头文件替换 |
| `include <emmintrin.h>` 编译失败 | 第2节 | 使用 SIMDE 头文件替换 |
| `include <arm_neon.h>` 编译失败 | 第2节 | 使用 SIMDE 头文件替换 |
| `__rdtsc` 未定义 | 第3.1节 | 替换为 `asm("rdtime.d %0":"=r"(val))` |
| `_mm_pause` 未定义 | 第3.2节 | 替换为空 |
| `_mm_clflush` / `_mm_sfence` / `_mm_mfence` 未定义 | 第3.3节 | 替换为 `asm volatile("dbar 0":::"memory")` |
| `mv %0, sp` 内联汇编错误 | 第3.4节 | 替换为 `addi.d %0, $sp, 0` |
| `CMAKE_SYSTEM_PROCESSOR` 判断缺失 | 第4.2节 | 添加 `loongarch64\|loong64` 匹配 |
| JIT/重定位/指令编码问题 | 第4.3节 | 参考 RISC-V64 实现和 GCC/LLVM LA64 后端 |
| glibc 地址模型假设错误 | 第4.1节 | LA64 使用绝对地址，行为同 X86-64 |

---

## 7. Python Copy-and-Patch JIT 移植指南

### 7.1 JIT 系统架构（Python 3.14）

Python 3.14 的 JIT 使用 Copy-and-Patch 技术（Brandon Lucia 和 Matthew T. Yourst 提出的方法）：

```
Hot Loop Trace (UOp IR)
    │
    ▼
Tier2 Optimizer ───▶ Optimized Bytecode Trace
    │                                      │
    │ ─── _PyJIT_Compile() ────────────────┘
    ▼
JIT Stencils (pre-compiled via clang)
    │
    ├─ Shim (entry point, standard CC)
    ├─ Stencil[0]_START_EXECUTOR
    ├─ Stencil[1]_LOAD_FAST
    ├─ Stencil[2]_BINARY_OP
    ├─ ...
    └─ Stencil[N]_FATAL_ERROR
    │
    ▼
Executable Code (mmap + copy + patch)
```

### 7.2 移植需要修改的文件

| 文件 | 修改内容 | 风险等级 |
|------|---------|---------|
| `Tools/jit/_targets.py` | 添加 LA64 target 的 triple 匹配和架构定义 | L0 |
| `Tools/jit/_stencils.py` | 添加 LA64 NOP 编码、对齐字节、Trampoline IR | L1 |
| `Tools/jit/_llvm.py` | 更新 LLVM 版本检测逻辑 | L0 |
| `Tools/jit/shim.c` | 调用链处理（防止尾调用优化跳过栈恢复） | L1 |
| `Python/jit.c` | 添加 LA64 重定位处理、补丁函数、ELFRunner 支持 | L2 |

### 7.3 LA64 重定位处理

需要支持的 LA64 核心重定位类型（从 binutils-2.45 `include/elf/loongarch.h` 提取）：

| 重定位类型 | 编号 | 指令 | 用途 |
|-----------|------|------|------|
| `R_LARCH_B16` | 64 | beq/bne/blt/bge/bltu/bgeu/jirl | 条件分支/jirl（16位有符号偏移） |
| `R_LARCH_B21` | 65 | beqz/bnez | 零条件分支（21位有符号偏移） |
| `R_LARCH_B26` | 66 | b/bl | 无条件跳转/函数调用（26位有符号偏移） |
| `R_LARCH_ABS_HI20` | 67 | lu12i.w | 绝对地址高20位（>>12） |
| `R_LARCH_ABS_LO12` | 68 | ori | 绝对地址低12位 |
| `R_LARCH_ABS64_LO20` | 69 | lu32i.d | 64位绝对地址第32-51位 |
| `R_LARCH_ABS64_HI12` | 70 | lu52i.d | 64位绝对地址第52-63位 |
| `R_LARCH_PCALA_HI20` | 71 | pcalau12i | PC相对地址高20位 |
| `R_LARCH_PCALA_LO12` | 72 | addi.w/addi.d | PC相对地址低12位 |
| `R_LARCH_PCALA64_LO20` | 73 | lu32i.d | 64位PC相对地址middle 20位 |
| `R_LARCH_PCALA64_HI12` | 74 | lu52i.d | 64位PC相对地址高12位 |
| `R_LARCH_GOT_PC_HI20` | 75 | pcalau12i | GOT PC相对高20位 |
| `R_LARCH_GOT_PC_LO12` | 76 | ld.w/ld.d | GOT PC相对低12位 |
| `R_LARCH_GOT64_PC_LO20` | 77 | lu32i.d | 64位GOT PC相对middle 20位 |
| `R_LARCH_GOT64_PC_HI12` | 78 | lu52i.d | 64位GOT PC相对高12位 |
| `R_LARCH_GOT_HI20` | 79 | lu12i.w | GOT绝对地址高20位 |
| `R_LARCH_GOT_LO12` | 80 | ori | GOT绝对地址低12位 |
| `R_LARCH_PCREL20_S2` | 103 | pcaddi | 短距PC相对（链接器relax后压缩pc_hi20+pc_lo12为单指令） |
| `R_LARCH_CALL36` | 110 | pcaddu18i + jirl | **中等代码模型函数调用**，两指令对36位偏移 |
| `R_LARCH_64_PCREL` | 109 | — | 64位PC相对（调试信息/eh_frame） |

> **注**：TLS 相关重定位类型（`R_LARCH_TLS_*`, 类型 6-14, 83-126）和链接器内部类型（`R_LARCH_RELAX`/`DELETE`/`ALIGN`, 100-102）、SOP 栈操作类型（22-46）、ADD/SUB 算术类型（47-56, 105-108）未在上表列出。完整列表见 binutils `include/elf/loongarch.h`（共 123 个类型）。

**补丁函数实现（关键修正）：**

> **⚠️ 重要：** 以下函数已根据 binutils-2.45.1 `bfd/elfxx-loongarch.c` 的 HOWTO 表和 `adjust_reloc_bits_field()` 编码逻辑进行**校验和修正**。之前的版本存在两处编码错误（B26 线性编码、si12 bit 11 移位错误），现已修复。

```c
// ─── 20-bit immediate patch (for pcaddu12i/pcalau12i/lu12i.w/lu32i.d) ───
// Fields used by: R_LARCH_PCALA_HI20, R_LARCH_GOT_PC_HI20, R_LARCH_ABS_HI20,
//                  R_LARCH_PCREL20_S2, R_LARCH_ABS64_LO20, etc.
// bitpos=5, bitsize=20, dst_mask=0x1ffffe0
static void patch_loongarch64_20r(unsigned char *code, uint32_t value) {
    uint32_t insn = *(uint32_t *)code;
    insn &= 0xfc00001f;   // clear bits [24:5]
    insn |= (value & 0xfffff) << 5;
    *(uint32_t *)code = insn;
}

// ─── 12-bit immediate patch (for addi.d/ld.d/ori/and/...) ───
// Fields used by: R_LARCH_PCALA_LO12, R_LARCH_GOT_PC_LO12, R_LARCH_ABS_LO12,
//                  R_LARCH_TLS_LE_LO12, etc.
// bitpos=10, bitsize=12, dst_mask=0x3ffc00
static void patch_loongarch64_12(unsigned char *code, uint32_t value) {
    uint32_t insn = *(uint32_t *)code;
    insn &= 0xffc003ff;   // clear bits [21:10]
    insn |= (value & 0xfff) << 10;
    *(uint32_t *)code = insn;
}

// ─── 26-bit branch offset patch (for b/bl) ───
// R_LARCH_B26: 无条件跳转，26 位有符号偏移
// ⚠️ 编码为分裂格式，非简单线性！
//   低 16 位 → 指令 bits[25:10]，高 10 位 → 指令 bits[9:0]
//   rightshift=2，即 offset = (target - PC) >> 2
static void patch_loongarch64_26r(unsigned char *code, int32_t offset) {
    uint32_t insn = *(uint32_t *)code;
    insn &= 0xfc000000;   // clear bits [25:0]
    // B26 split encoding: lo16→bits[25:10], hi10→bits[9:0]
    uint32_t lo16 = (offset & 0xffff) << 10;
    uint32_t hi10 = (offset >> 16) & 0x3ff;
    insn |= lo16 | hi10;
    *(uint32_t *)code = insn;
}

// ─── 36-bit call patch (for pcaddu18i + jirl pair, medium code model) ───
// R_LARCH_CALL36: 占用两个连续的 4 字节指令空间
//   低 16 位（经符号扩展补偿后）→ 第二指令 bits[25:10]
//   高位（16+2 位右移后）→ 第一指令 bits[4:0]
//   rightshift=2, size=8
static void patch_loongarch64_call36(unsigned char *code, int64_t offset) {
    uint32_t *insn = (uint32_t *)code;
    // 符号扩展补偿（若低16位为负，高20位需 +0x8000）
    int64_t adj = offset + 0x8000;
    // 第一指令 (pcaddu18i): 高18位 → bits[4:0]
    insn[0] = (insn[0] & 0xfc00001f) | (((adj >> 16) & 0x7ffff) << 5);
    // 第二指令 (jirl): 低16位 → bits[25:10]
    insn[1] = (insn[1] & 0xfc0003ff) | (((offset & 0xffff) << 10));
}
```

### 7.3.1 编码细节说明

**B26 分裂编码原理**

B26 的 26 位偏移量在 B/BL 指令中分两段存储（参考 binutils `adjust_reloc_bits_field`）：
```
val:          [25........................0]
指令编码:      [25:10]=val[15:0]  [9:0]=val[25:16]
               高位 16 bit            低位 10 bit
```

这是因为 B 指令格式中，26 位立即数被拆分为两个不连续字段。简单的 `insn |= val` 无法正确编码。

**CALL36 两指令对**

R_LARCH_CALL36 覆盖两个相邻指令（共 8 字节），用于中等代码模型（±128GB 范围）：
```
insn[0] pcaddu18i rd, imm20    ← hi18 = (offset + 0x8000) >> 16
insn[1] jirl    rd, rj, imm16  ← lo16 = offset & 0xFFFF
```
符号扩展补偿 `+0x8000`：当低 16 位为负时（bit 15=1），高 18 位需 +1 补偿。

**PCREL20_S2 单指令优化**

当 `pc_hi20 + pc_lo12` 对的目标在 ±1MB 范围内，链接器可将其 relax 为单条 `pcaddi` 指令（R_LARCH_PCREL20_S2）：
```
pcaddu12i + addi.d  →  pcaddi   (relax 后)
20bit >>12 + 12bit  →  20bit>>2  (对齐要求从页变为4字节)
```

### 7.3.2 静态链接 vs 动态链接重定位分类

| 范畴 | 类型编号范围 | 典型使用 |
|------|------------|---------|
| 动态链接器 | 0-14 | `R_LARCH_RELATIVE`, `R_LARCH_JUMP_SLOT`, TLS 等 |
| 静态链接 (.text) | 20-103, 109-110 | 上表中的指令 patch 类型 |
| 静态链接 (非.text) | 47-58 | `R_LARCH_ADD32`, `R_LARCH_SUB32` 等数据段 |
| 链接器 Relax | 100-102 | `R_LARCH_RELAX`, `R_LARCH_DELETE`, `R_LARCH_ALIGN` |
| TLS 完整族 | 6-14, 83-98, 111-126 | LE/IE/LD/GD/DESC 各模型 |

### 7.4 调用约定边界处理

**核心问题：** JIT 编译的 stencil 使用 `preserve_none` CC（LLVM 专有属性），但入口点由 GCC 编译的 CPython 解释器调用。GCC 不了解 `preserve_none`，会预期 callee-saved 寄存器被保存。

**解决方案——标准CC入口 Shim：**

```c
// shim.c — 标准 CC 入口点
_Py_CODEUNIT *
_JIT_ENTRY(_PyInterpreterFrame *frame, _PyStackRef *sp, PyThreadState *tstate)
{
    PATCH_VALUE(jit_func_preserve_none, call, _JIT_CONTINUE);
    // volatile 防止编译器优化为尾调用（跳过栈恢复）
    jit_func_preserve_none volatile call_volatile = call;
    return call_volatile(frame, sp, tstate);
}
```

`volatile` 关键字强制编译器生成常规调用指令（`bl` + `ret`），而非尾部调用（`b`），从而确保 shim 的栈帧寄存器的保存/恢复序列正确执行。

### 7.5 测试验证

| 模式 | 预期行为 |
|------|---------|
| `PYTHON_JIT=0` | 完全禁用 tier2 和 JIT，基础 Python 正常运行 |
| `PYTHON_JIT=yes` | JIT 编译热循环，但 tier2 字节码执行器有已知 LA64 bug |
| `PYTHON_JIT=no` | 仅 tier2 优化（不编译 JIT），与 yes 模式有相同的 tier2 bug |

> **已知问题：** Python 3.14 的 tier2 字节码执行器在 LA64 上存在 bug（在 ~5000 次迭代后崩溃），发生在 `Python/executor_cases.c.h`（生成的文件）中。此问题与 JIT 移植无关，在未修改的源码中也会复现。

---

## 8. 移植修复模板

### 8.1 条件编译标准模板

```c
#if defined(__loongarch_lp64)
    // LoongArch64 特定实现
#elif defined(__x86_64__) || defined(__amd64__)
    // X86-64 原始实现
#elif defined(__aarch64__)
    // ARM64 实现
#else
#  error "Unsupported architecture"
#endif
```

### 8.2 CMake 架构检测模板

```cmake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|loong64")
    set(ARCH_LOONGARCH64 TRUE)
    add_definitions(-DARCH_LOONGARCH64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_AARCH64 TRUE)
endif()
```

### 8.3 Autotools 架构检测模板

```m4
# 在 configure.ac 中添加：
case "$host_cpu" in
    loongarch64)
        ARCH_LOONGARCH64=yes
        AC_DEFINE([ARCH_LOONGARCH64], [1], [LoongArch64 architecture])
        ;;
    x86_64|amd64)
        ARCH_X86_64=yes
        ;;
esac
```

---

## 9. rpmbuild 与 spec 文件编写

### 9.1 概述

`rpmbuild` 是 RHEL/Fedora/CentOS/Rocky/AlmaLinux 等 RPM 系发行版的标准包构建工具。一个 `.spec` 文件描述了如何从源码构建出一个或多个 RPM 包，包括依赖声明、构建步骤、文件清单和安装脚本。

对于 LoongArch64 移植工作，rpmbuild 提供两大价值：
1. **构建信息提取** — spec 文件精确记录依赖、构建命令、测试命令，是比 README 更权威的构建参考
2. **可复现构建** — 使用 `mock` 在干净 chroot 中构建，保证环境一致性，避免"在我机器上能编"的问题

### 9.2 spec 文件结构速查

```spec
# ─── 包元信息（Preamble） ───
Name:           example
Version:        1.2.3
Release:        1%{?dist}
Summary:        Example package for LoongArch64

License:        MIT
URL:            https://example.com/project
Source0:        %{url}/archive/v%{version}.tar.gz

# 架构声明：明确列出支持的架构
# 如需声明仅支持特定架构，使用 ExclusiveArch
ExclusiveArch:  x86_64 aarch64 loongarch64

# 构建依赖（编译时需要的包）
BuildRequires:  gcc
BuildRequires:  make
BuildRequires:  cmake
BuildRequires:  pkgconfig(zlib)

# 运行时依赖
Requires:       zlib

%description
Example package ported to LoongArch64.

# ─── 构建阶段 ───
%prep
# 解压源码并进入源码目录
%autosetup

%build
# 编译命令
%configure
%make_build

%install
# 安装到虚拟根目录 %{buildroot}
%make_install

%check
# 测试命令
make test

# ─── 文件清单 ───
%files
%license LICENSE
%doc README.md
%{_bindir}/example
%{_libdir}/libexample.so.*

%changelog
* Mon Jun 02 2026 LoongArch Porter <porter@example.com> - 1.2.3-1
- Initial LoongArch64 port
```

### 9.3 常用 rpm 宏速查

| 宏 | 展开值（典型） | 用途 |
|----|--------------|------|
| `%{_bindir}` | `/usr/bin` | 可执行文件安装路径 |
| `%{_libdir}` | `/usr/lib64` (64位) | 库文件路径 |
| `%{_datadir}` | `/usr/share` | 数据文件路径 |
| `%{_includedir}` | `/usr/include` | 头文件路径 |
| `%{buildroot}` | `~/rpmbuild/BUILDROOT/%{name}-%{version}-%{release}.%{_arch}` | 安装虚拟根目录 |
| `%{_smp_mflags}` | `-j$(nproc)` | 并行编译标志 |
| `%{_arch}` | `loongarch64` (在 LA64 上) | 当前构建架构 |
| `%{?dist}` | `.el9` / `.fc40` 等 | 发行版标识 |
| `%{?_with_foo}` | `--with-foo` 或空 | 条件构建选项 |

### 9.4 为 LoongArch64 适配 spec 文件

#### 9.4.1 架构声明

RPM 原生支持 LoongArch64 架构名。按以下方式声明：

```spec
# 方式 A：除特定架构外均可构建（如果项目源码本身架构无关）
# 不设置 ExclusiveArch 或使用：
ExclusiveArch:  %{ix86} x86_64 aarch64 loongarch64

# 方式 B：明确允许所有架构（架构无关包如 Python/Perl 模块）
BuildArch:      noarch

# 方式 C：仅允许特定架构（如包含大量内联汇编的项目）
ExclusiveArch:  x86_64 aarch64 loongarch64
```

#### 9.4.2 架构条件编译

```spec
%build
# 根据目标架构选择不同的编译选项
%ifarch loongarch64
    %configure --with-arch=loongarch64 --disable-simd-native
%endif
%ifarch x86_64
    %configure --with-arch=x86_64 --enable-avx2
%endif
%ifarch aarch64
    %configure --with-arch=aarch64 --enable-neon
%endif
%make_build
```

#### 9.4.3 架构条件依赖

```spec
# 仅在 LA64 上需要的依赖
%ifarch loongarch64
BuildRequires:  simde-devel
%endif

# 仅在 X86 上需要的依赖
%ifarch x86_64
BuildRequires:  nasm
%endif
```

#### 9.4.4 架构条件文件清单

```spec
%files
%{_bindir}/example

# 仅在 LA64 上安装的文件
%ifarch loongarch64
%{_libdir}/libexample_la64.so
%endif

# 仅在 X86 上安装的文件
%ifarch x86_64
%{_libdir}/libexample_x86.so
%endif
```

### 9.5 rpmbuild 工作流

**步骤 1：设置构建目录**

```bash
# 安装必要工具
sudo dnf install rpmdevtools rpmlint
# 创建 ~/rpmbuild/ 目录树（BUILD, RPMS, SOURCES, SPECS, SRPMS）
rpmdev-setuptree
```

**步骤 2：放置源码和 spec**

```bash
# 将源码 tarball 放入 SOURCES 目录
cp example-1.2.3.tar.gz ~/rpmbuild/SOURCES/
# 将 spec 文件放入 SPECS 目录
cp example.spec ~/rpmbuild/SPECS/
```

**步骤 3：安装构建依赖**

```bash
# 自动安装 spec 中声明的所有 BuildRequires
sudo dnf builddep ~/rpmbuild/SPECS/example.spec
```

**步骤 4：构建**

```bash
# 构建二进制包（-ba 同时生成源码包 SRPM）
rpmbuild -ba ~/rpmbuild/SPECS/example.spec
# 仅构建二进制包
rpmbuild -bb ~/rpmbuild/SPECS/example.spec
# 仅生成源码包（用于在 LA64 机器上重新构建）
rpmbuild -bs ~/rpmbuild/SPECS/example.spec
```

**步骤 5：验证**

```bash
# 使用 rpmlint 检查 spec 和生成的 RPM 是否符合规范
rpmlint ~/rpmbuild/SPECS/example.spec
rpmlint ~/rpmbuild/RPMS/loongarch64/example-*.rpm
```

### 9.6 使用 mock 进行干净 chroot 构建

`mock` 工具在隔离的 chroot 环境中构建，确保不依赖宿主机环境。对于移植工作尤为重要——可让 X86-64 机器借助 QEMU 构建 LA64 包。

```bash
# 安装 mock
sudo dnf install mock

# 将自己加入 mock 组
sudo usermod -a -G mock $USER

# 初始化 LA64 架构的 chroot（需 LoongArch64 发行版配置）
mock --init -r loongarch64-koji-config

# 构建 SRPM
mock --buildsrpm --spec example.spec --sources ~/rpmbuild/SOURCES/ \
     -r loongarch64-koji-config

# 构建二进制包
mock --rebuild /var/lib/mock/loongarch64-koji-config/result/example-*.src.rpm \
     -r loongarch64-koji-config
```

**跨架构 mock 使用场景**：

| 场景 | 宿主机架构 | mock 目标架构 | 机制 |
|------|-----------|-------------|------|
| LA64 原生构建 | loongarch64 | loongarch64 | 原生 chroot，最快 |
| X86→LA64 交叉 | x86_64 | loongarch64 | QEMU 用户态模拟 |
| 参考构建 | x86_64 | x86_64 | 原生 chroot |

### 9.7 从现有 spec 提取移植信息

当遇到一个已有 spec 文件的项目，Agent 应提取以下关键字段：

```bash
# 快速提取关键信息（不依赖 rpmspec 解析器时）
grep -E '^(Name|Version|BuildRequires|Requires|ExclusiveArch):' example.spec
```

| 字段 | 移植用途 |
|------|---------|
| `Name` / `Version` | 确认项目标识和版本 |
| `BuildRequires` | **最重要**——知道编译需要什么，对比 LA64 是否可用 |
| `ExclusiveArch` | 检查是否已声明 LA64 支持 |
| `%build` 段 | 提取准确编译命令和选项 |
| `%check` 段 | 提取测试命令 |
| `%install` 段 | 了解哪些文件是构建产物 |
| `%ifarch` 块 | 发现已有架构特殊处理，评估 LA64 适配位置 |

### 9.8 常见构建失败与修复

| 错误现象 | 原因 | 修复 |
|---------|------|------|
| `error: Architecture is not included: loongarch64` | spec 的 `ExclusiveArch` 不含 loongarch64 | 将 `loongarch64` 添加到 `ExclusiveArch` 列表 |
| `error: Failed build dependencies: foo is needed by ...` | LA64 上缺少某个依赖包 | 确认依赖在 LA64 上是否可用，若不可用需先移植依赖 |
| `configure: error: unsupported architecture` | 项目的 configure 不认识 loongarch64 | 按本指南 §1 处理 autotools |
| `gcc: error: unrecognized command line option '-msse2'` | spec 中硬编码了 X86 特有编译选项 | 使用 `%ifarch` 隔离架构选项 |
| `mock: no configuration for loongarch64` | mock 没有 LA64 的配置文件 | 创建 `/etc/mock/loongarch64.cfg`，参考发行版文档 |

---

> **维护记录**
>
> | 日期 | 更新内容 | 更新者 |
> |------|----------|--------|
> | 2026-05-13 | 初始化，填入autotools处理、SIMDE替换、特定函数映射、平台特性 | Agent (Kimi) |
> | 2026-06-04 | 新增 §9 rpmbuild/spec 编写章节 | Hermes Agent |
