# 常见陷阱与教训：LoongArch64 软件移植

> **Agent 阅读指引**
> 
> 本文档记录OpenClaw移植项目中Agent容易犯的错误，尤其是跨架构移植特有的陷阱。
> 
> 在执行类似移植任务前，请先查阅本节，避免重复犯错。

## 1. 移植流程陷阱

### 陷阱 1.1：跳过X86-64参考构建

**问题**：用户催促或Agent自信，直接进入LoongArch64移植，发现构建失败，却无法判断是项目本身的问题还是架构差异。

**后果**：
- 浪费时间在修复项目本身的通用Bug上
- 无法区分"移植问题"和"项目问题"
- 无法验证移植后的行为是否与X86-64一致

**预防措施**：
- **严格执行阶段2**：任何移植必须先完成X86-64参考构建
- 记录X86-64的完整构建日志和测试输出
- 如果X86-64构建失败，先解决后再移植

---

### 陷阱 1.2：忘记初始提交

**问题**：直接修改源码后才发现没有Git仓库，无法区分原始代码和修改。

**后果**：
- 无法生成干净的移植补丁
- 无法回溯原始状态
- 可能丢失修改

**预防措施**：
- **强制检查**：每次处理新项目，第一步检查 `.git` 目录
- 使用检查脚本（见 `operations/safe-operations.md` 第7节）

---

### 陷阱 1.3：构建产物跨架构污染

**问题**：在X86-64上构建了项目，然后直接在LoongArch64上继续构建，没有清理build目录。

**后果**：
- 链接器可能混用X86-64和LoongArch64的目标文件
- 出现难以理解的链接错误
- 可能链接出无法运行的混合架构二进制

**预防措施**：
- **切换架构时清理build目录**：`rm -rf build/ && mkdir build`
- 使用 `file` 命令验证构建产物的架构：`file build/src/some_binary`
- 推荐在CMake/Make中使用独立目录：`-B build-x86_64` 和 `-B build-loongarch64`

---

## 2. 源码修改陷阱

### 陷阱 2.1：破坏X86-64兼容性

**问题**：为适配LoongArch64修改了通用代码，导致X86-64上无法编译或行为改变。

**案例**：
```c
// 错误：直接修改通用代码
int pointer_size = 8;  // 原来 sizeof(void*)

// 正确：使用条件编译
#ifdef __loongarch__
// LA64特定代码
#else
// 其他架构代码
#endif
```

**预防措施**：
- 使用条件编译隔离架构差异：`#ifdef __loongarch__` / `#if defined(__loongarch64)`
- 修改后回到X86-64验证编译（如可能）
- 优先使用标准C/C++类型（`uintptr_t`, `size_t`）而非假设指针宽度

---

### 陷阱 2.2：忽视字节序差异

**问题**：LoongArch64可以配置为大端或小端，但通常为小端（与X86-64相同）。如果代码假设了字节序，可能在特定配置下出错。

**预防措施**：
- 如果项目已有字节序处理，确认是否覆盖LoongArch64
- 使用标准宏检测：`#include <endian.h>` 后使用 `__BYTE_ORDER__`
- 不要假设LoongArch64的字节序

---

### 陷阱 2.3：内联汇编硬编码

**问题**：项目包含X86-64特有的内联汇编（SSE/AVX指令、CPUID等），直接移植到LoongArch64。

**常见场景**：
- `cpuid` 指令获取CPU信息 → 需替换为LA64等价物或系统调用
- `rdtsc` 读取时间戳 → 替换为LA64 `rdtime` 或标准 `clock_gettime`
- 原子操作汇编 → 替换为C11 `_Atomic` 或GCC内置函数

**预防措施**：
- 查找所有 `__asm__`, `asm()`, `__asm` 出现位置
- 优先考虑使用编译器内置函数而非内联汇编
- 如果没有LA64等价物，使用条件编译提供回退实现

---

### 陷阱 2.4：LA64 指令编码假设（从 AArch64 移植）

**问题**：将 AArch64 的指令编码逻辑直接搬到 LA64。

**典型场景**：

1. **`ADRP` → `PCADDU12I` 假页对齐**
   - AArch64 的 `ADRP` 会清除 PC 的低12位（页对齐），然后 `LDR` 用 `target & 0xFFF` 作为偏移
   - LA64 的 `PCADDU12I` **不做页对齐**，PC 的低12位保留
   - 错误做法：`imm20 = (page_of_target − page_of_PC) >> 12, imm12 = target & 0xFFF`
   - 正确做法：`diff = target − PC, imm20 = diff >> 12, si12 = diff & 0xFFF`（处理符号扩展）

2. **`ALSL.D` 的移位量**
   - `ALSL.D rd, rj, rk, sa` 计算 `rd = rk + (rj << sa)`
   - `sa` 是**直接移位量**，不是 `sa+1`
   - 验证：`alsl.d $a0, $a1, $a2, 1` = `$a0 = $a2 + ($a1 << 1)`

### 陷阱 2.5：指针与整数转换

**问题**：假设 `sizeof(long) == sizeof(void*)`，在ILP32和LP64模型中不同。

| 数据模型 | int | long | long long | pointer |
|----------|-----|------|-----------|---------|
| X86-64 LP64 | 32 | 64 | 64 | 64 |
| LoongArch64 LP64 | 32 | 64 | 64 | 64 |
| X86 ILP32 | 32 | 32 | 64 | 32 |

**预防措施**：
- 使用 `<stdint.h>` / `<cstdint>` 中的定宽类型
- 指针转整数用 `uintptr_t`，整数转指针用 `void*` 转换
- 不要假设 `long` 的宽度

---

## 3. 构建系统陷阱

### 陷阱 3.1：架构检测错误

**问题**：构建系统使用 `uname -m` 或自定义脚本来检测架构，但不认识 `loongarch64`。

**常见症状**：
- `./configure` 报错："Unknown architecture"
- CMake 选择了错误的编译选项
- Makefile 中没有为loongarch64定义的条件分支

**修复方法**：
- autotools: 在 `configure.ac` 中添加 `loongarch64` 到架构列表
- CMake: 检查 `CMAKE_SYSTEM_PROCESSOR` 的处理逻辑
- 手动Makefile: 添加 `ifeq ($(ARCH),loongarch64)` 分支

---

### 陷阱 3.2：依赖检测失败

**问题**：LoongArch64环境下缺少某些依赖库，构建系统检测不到。

**预防措施**：
- 参考ebuild/PKGBUILD中的依赖列表，在LA64上逐一确认
- 使用发行版包管理器安装缺失依赖
- 记录LA64特有的依赖安装命令

---

### 陷阱 3.3：编译器标志不兼容

**问题**：项目使用了GCC特有的标志（如 `-march=haswell`），在LoongArch64 GCC上不识别。

**预防措施**：
- 检查 `-march=`, `-mtune=`, `-msse*` 等X86特有标志
- 为LoongArch64替换为 `-march=loongarch64`, `-mtune=la464` 等
- 使用条件编译而非全局编译标志（如可能）

---

## 4. 测试验证陷阱

### 陷阱 4.1：X86-64测试本身就失败

**问题**：移植完成后，对比X86-64和LoongArch64的测试结果，但发现X86-64的某些测试本来就失败。

**后果**：
- 错误地将X86-64的已知失败归因于LoongArch64移植
- 浪费时间在修复非移植相关的问题上

**预防措施**：
- 记录X86-64参考构建的完整测试结果
- 区分"X86-64已知失败"和"LoongArch64新增失败"
- 移植报告中明确标注："X86-64基线：X个测试失败，LoongArch64：Y个新增失败"

---

### 陷阱 4.2：仅验证编译通过

**问题**：移植后只在LoongArch64上编译通过，但没有运行测试套件。

**后果**：
- 运行时逻辑错误未被发现
- 测试用例可能在运行时失败

**预防措施**：
- **编译通过不等于移植完成**
- 必须运行与X86-64相同的测试套件
- 如果测试运行缓慢（如QEMU模拟），至少运行核心功能测试

---

## 5. Agent 行为陷阱

### 陷阱 5.1：过度自信——"这个改动很小"

**问题**：Agent认为某个修改"只是添加了一个宏"，但实际上影响了全局编译流程。

**教训**：在跨架构移植中，没有"小"改动。每个修改都可能影响整个构建系统。

### 陷阱 5.2：望文生义理解构建系统

**问题**：看到 `Makefile` 就认为可以直接 `make`，没看到项目需要 `autoreconf` 或 `cmake` 预处理。

**教训**：
- 始终先阅读构建文档（README/docs/build.md）
- 检查是否存在 `configure.ac`（需要autotools）或 `CMakeLists.txt`（需要cmake）
- 不要盲目执行命令

### 陷阱 5.3：忽视构建日志

**问题**：构建失败后只看最后一行错误，没有阅读完整日志。

**教训**：
- 移植相关的错误可能在日志的早期就出现（如配置阶段）
- 使用 `make 2>&1 | tee build.log` 保存完整日志
- 分析错误链，不只是最后一个错误

### 陷阱 5.4：未记录移植步骤

**问题**：Agent解决了构建问题，但没有记录具体做了什么，导致后续无法复现。

**教训**：
- 每个解决问题的步骤都应记录
- 保存构建日志和错误信息
- 最终生成可复现的移植指南

---

## 6. SIMDE 相关陷阱

### 陷阱 6.1：遗漏 SIMDE 依赖

**问题**：使用 SIMDE 头文件替换了 X86 SIMD 头文件，但系统未安装 SIMDE 包，导致编译仍然失败。

**预防措施**：
- 在使用 SIMDE 前，先检查系统是否已安装：
  ```bash
  pkg-config --exists simde && echo "SIMDE available" || echo "SIMDE not found"
  # 或检查头文件
  ls /usr/include/simde/
  ```
- 如果未安装，记录为依赖并安装（或通过发行版包管理器安装）

### 陷阱 6.2：SIMDE 头文件路径错误

**问题**：使用了错误的 SIMDE 头文件路径，如 `<simde/x86/avx2.h>` 写成了 `<simde/x86/avx.h>`。

**预防措施**：
- 严格参照 `la64-porting-guide.md` 第2.3节的映射表
- 如果SIMDE版本较旧，某些头文件可能不存在，需确认版本兼容性

### 陷阱 6.3：`_mm_pause` 替换后引发性能问题

**问题**：将 `_mm_pause()` 替换为空后，自旋锁在空闲时占用100% CPU。

**预防措施**：
- 在文档中明确说明：`_mm_pause()` 在 LA64 上无等价物
- 如果项目性能敏感，建议使用操作系统提供的同步原语（如 `futex`）替代忙等待
- 在移植报告中注明此性能差异

---

## 7. Autotools 相关陷阱

### 陷阱 7.1：覆盖 config.sub 后未重新生成 configure

**问题**：覆盖了 `config.sub` 和 `config.guess`，但没有重新运行 `autoreconf`/`bootstrap`，导致 configure 仍然不认识 loongarch64。

**预防措施**：
- 严格遵循 `la64-porting-guide.md` 第1.2节的流程：先覆盖 config 脚本，再重新生成 configure
- 可通过 `./configure --help | grep host` 检查是否支持 `--host=loongarch64-linux-gnu`

### 陷阱 7.2：autoreconf 失败

**问题**：执行 `autoreconf -fvis` 时因缺少依赖（如 `aclocal`, `libtool`）而失败。

**预防措施**：
- 先确保 autotools 完整安装：
  ```bash
  autoconf --version
  automake --version
  libtool --version
  ```
- 在 ebuild/PKGBUILD 中查找 BDEPEND/makedepends 确认构建依赖
- 在移植报告中记录需要的额外构建依赖

### 陷阱 7.3：系统 config.sub 版本过旧

**问题**：从 `/usr/share/` 找到的 `config.sub` 本身就不支持 loongarch64。

**预防措施**：
- 覆盖后验证：`./config.sub loongarch64-unknown-linux-gnu`
- 如果输出不是 `loongarch64-unknown-linux-gnu`，说明系统版本过旧
- 从最新的 GNU config 仓库获取：https://git.savannah.gnu.org/cgit/config.git/plain/config.sub
- 或使用 `autoreconf -fvis` 时指定 `--install` 自动下载最新版本

---

## 8. 历史事件记录

> 每次Agent在移植过程中导致或差点导致问题时，在此记录，供后续Agent学习。

| 日期 | 项目 | 问题描述 | 根本原因 | 预防措施 |
|------|------|----------|----------|----------|
| 2026-06-03 | Python 3.14 JIT | LLVM 18 缺少 `preserve_none` 属性，stencil 函数始终分配栈帧，JIT 的 musttail 链中栈不断增长 | LLVM 18 的 LA64 后端不支持 `preserve_none` 取消调用者保存约定 | 需使用 LLVM 19+，并为 LA64 后端手动打补丁添加 `CSR_NoneRegs` |
| 2026-06-03 | Python 3.14 JIT | Shim 入口点使用 `__attribute__((preserve_none))` 后，从 GCC 编译的解释器调用时 callee-saved 寄存器被污染 | GCC 不了解 `preserve_none`，expects 标准调用约定；加了 preserve_none 后 shim 不保存寄存器 | Shim 必须使用标准 CC；使用 `volatile` 关键字防止编译器将 `return call()` 优化为尾调用 |
| 2026-06-03 | Python 3.14 JIT | LLVM 19 `CSR_NoRegs`（空保存列表）用于 preserve_none 时，$ra 未保存导致返回错误 | preserve_none 不保存任何寄存器，包括 $ra。在 musttail 链中 $ra 虽然稳定，但入口的 bl 调用已修改 $ra | 使用 `CSR_NoneRegs`（R1, R22）替代 `CSR_NoRegs`，在入口/出口处保存/恢复 $ra 和 $fp |
| 2026-06-03 | Python 3.14 JIT | LA64 的 `pcaddu12i` 指令不像 AArch64 的 `adrp` 做 PC 页对齐，导致所有 GOT 加载计算了错误的地址 | 移植 `patch_loongarch64_20r` 和 `patch_loongarch64_12` 时错误地拷贝了 AArch64 的页对齐行为 `(target & ~0xFFF)` | 使用 `diff = target - PC` 的正确拆分：`imm20 = diff >> 12`，`si12 = diff & 0xFFF`（并处理 12 位符号扩展） |
| 2026-06-03 | Python 3.14 JIT | Trace 退出时 `_ERROR_POP_N` 返回 NULL，GOTO_TIER_TWO 的 `instr_ptr + 1` 导致解释器跳过了正确的退出指令 | `_ERROR_POP_N` 已设置 `frame->instr_ptr` 为正确地址，但返回 NULL 触发 GOTO_TIER_TWO 的 +1 补偿，在 LA64 上算错了位置 | 修改 `_ERROR_POP_N` 返回计算出的 `instr_ptr` 值（而非 NULL），使 GOTO_TIER_TWO 直接分发 |
| 2026-06-03 | Python 3.14 JIT | `_GUARD_NOT_EXHAUSTED_RANGE` 的 JUMP_TARGET 分支跳转到 `_FATAL_ERROR`，而不是正确的退出 stencil | trace 退出控制流异常，疑似 `instruction_starts` 数组在特定条件下计算出错 | 待排查：可能是 `_GUARD` 的 `jump_target` 字段在每次循环迭代中被修改 |
| 2026-06-04 | LA64 重定位补丁 | `patch_loongarch64_26r` 使用 `(offset >> 2) & 0x03ffffff` 线性编码 B26，导致 B/BL 指令跳转目标错误 | B/BL 指令的 26 位立即数为**分裂编码**：bits[15:0]→指令bits[25:10], bits[25:16]→指令bits[9:0]，非连续字段 | 按 binutils `adjust_reloc_bits_field` 实现：`lo16 = (val & 0xffff) << 10; hi10 = (val >> 16) & 0x3ff` |
| 2026-06-04 | LA64 重定位补丁 | `patch_loongarch64_12` 中用 `(value & 0x800) ? 0x20000000` 将 si12 bit 11 放在了 bit 29 位置 | ADDI.D/LD.D 的 si12 字段占据 bits[21:10]，bit 11 应对应指令 bit 21 (= 0x200000)，而非 bit 29 | 正确实现：统一使用 `(value & 0xfff) << 10` 将全部 12 位置于 bits[21:10] |


---

## 9. 快速自检清单（移植完成后）

在报告移植完成前，检查：

- [ ] 已执行初始提交（`git init && git add -A && git commit -m "Initial import"`）
- [ ] 已完成X86-64参考构建并记录
- [ ] LoongArch64构建产物已验证架构（`file` 命令）
- [ ] 移植修改使用条件编译隔离（未破坏X86-64兼容性）
- [ ] 已运行测试套件并记录结果
- [ ] 已生成干净的移植补丁（`git diff` 或 `git format-patch`）
- [ ] 没有提交构建产物到Git
- [ ] 没有执行L0禁止操作
- [ ] 移植步骤已记录，可供他人复现
- [ ] **(autotools项目)** config.sub/guess 已更新且 configure 已重新生成
- [ ] **(SIMDE项目)** SIMDE依赖已确认可用，头文件路径正确
- [ ] **(CMake项目)** SYSTEM_PROCESSOR 已处理 loongarch64/loong64
- [ ] **(JIT项目)** LLVM 版本满足 preserve_none 要求（≥19 + 补丁）
- [ ] **(JIT项目)** 入口点调用约定 shim 已正确处理（标准 CC + 防尾调优化）
- [ ] **(JIT项目)** LA64 重定位类型已添加（R_LARCH_B26, GOT_PC_HI20/LO12, PCALA_HI20/LO12）
- [ ] **(JIT项目)** Tier2 字节码执行器是否有 LA64 bug（如 Python 3.14 的已知问题）
- [ ] **(RPM项目)** spec 文件 ExclusiveArch 已添加 loongarch64
- [ ] **(RPM项目)** 架构条件编译（%ifarch loongarch64）已处理
- [ ] **(RPM项目)** mock chroot 配置已就绪且 loongarch64 目标可用

---

## 9. rpmbuild / spec 相关陷阱

### 陷阱 9.1：遗忘 ExclusiveArch 声明

**问题**：原有 spec 的 `ExclusiveArch` 如 `x86_64 aarch64` 未包含 `loongarch64`，在 LA64 上 rpmbuild 直接拒绝构建。

**预防措施**：
- 始终检查 spec 的 `ExclusiveArch` 行
- 如果项目架构无关（Python/Perl/Shell），直接设为 `BuildArch: noarch`
- 若需限制架构，添加 `loongarch64` 到列表

### 陷阱 9.2：spec 中硬编码 X86 编译选项

**问题**：spec 的 `%build` 段包含 `-msse2 -mavx2` 等 X86 特有标志，在 LA64 上 GCC 报错。

**预防措施**：
- 使用 `%ifarch` 隔离架构编译选项
- LA64 通用选项：`-march=loongarch64 -mtune=la464`
- 参考 `la64-porting-guide.md` §9.4.2 的条件编译模板

### 陷阱 9.3：mock 缺少 LoongArch64 配置

**问题**：执行 `mock -r loongarch64-config` 时提示无此配置。

**预防措施**：
- 确认 `/etc/mock/` 下有 la64 的 `.cfg` 文件
- LA64 发行版（如 Loongnix）通常自带 mock 配置
- 若无，参考发行版文档创建，关键字段：`config_opts['target_arch'] = 'loongarch64'`

### 陷阱 9.4：忽视 %check 段测试

**问题**：spec 有 `%check` 段但 rpmbuild 默认可能跳过（需 `--with check` 或 spec 内 `%bcond_without check`）。

**预防措施**：
- 构建时使用 `rpmbuild -ba --with check` 强制执行测试
- 在 %check 段开头检查 `%{with check}` 条件

### 陷阱 9.5：BuildRequires 在 LA64 上不可用

**问题**：spec 的 `BuildRequires` 包含仅在 X86 上存在的包，`dnf builddep` 失败。

**预防措施**：
- 逐个检查 `BuildRequires` 在 LA64 发行版上的可用性
- 使用 `%ifarch` 隔离架构特定依赖
- 对于通用的开发库（如 `zlib-devel`），通常 LA64 已有

---

> **维护记录**
>
> | 日期 | 更新内容 | 更新者 |
> |------|----------|--------|
> | 2026-05-13 | 初始化并填入LoongArch64移植常见陷阱 | Agent (Kimi) |
> | 2026-06-04 | 新增 §9 rpmbuild/spec 陷阱，补充 RPM 检查清单 | Hermes Agent |
