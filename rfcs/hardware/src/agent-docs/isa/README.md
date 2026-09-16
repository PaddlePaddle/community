# ISA 模型文档中心

> 本目录包含 LoongArch64 与 RISC-V64 的完整 ISA YAML 模型，用于指导 Agent 在代码移植、翻译、优化时的架构决策。
>
> **数据权威性**：YAML 文件为原始机器可读模型，本文档为人工索引与使用指南。当两者冲突时，以 YAML 为准。

## 目录结构

```
isa/
├── README.md              # 本文档：使用指南与索引
├── ARCH_DIFF.md           # LA64 vs RV64 核心差异速查
├── loongarch64/
│   ├── _meta.yml          # 架构元信息：寄存器、ABI、CSR、LBT
│   ├── _formats.yml       # 指令格式模板（DJK/DJ/DI20 等）
│   ├── base.yml           # 基础整数指令（含编码与部分语义）
│   ├── base_semantics.yml # 基础指令语义补充
│   ├── missing_base.yml   # 基础指令中易遗漏的别名/扩展指令
│   ├── missing_base_semantics.yml
│   ├── bitops.yml         # 位操作指令（CLZ/CTZ/REVB/BSTRINS 等）
│   ├── mul.yml            # 乘除法指令
│   ├── atomics.yml        # 原子操作（LL/SC/AM*）
│   ├── atomics_semantics.yml
│   ├── fp-common.yml      # 浮点公共部分
│   ├── fp-s.yml / fp-d.yml# 单/双精度浮点
│   ├── bound.yml / bound_semantics.yml  # 边界检查指令
│   ├── lsx.yml / lsx_semantics.yml      # 128-bit SIMD (LSX)
│   └── lasx.yml / lasx_semantics.yml    # 256-bit SIMD (LASX)
└── riscv64/
    ├── _meta.yml          # 架构元信息：寄存器、ABI、CSR
    ├── _formats.yml       # 指令格式模板（R/I/S/B/U/J/C/V 等）
    ├── base.yml           # RV64I 基础整数指令
    ├── base_semantics.yml
    ├── rv64i-specific.yml / _semantics.yml   # RV64 特有指令
    ├── m-extension.yml / _semantics.yml      # 乘除法扩展
    ├── a-extension.yml / _semantics.yml      # 原子操作扩展
    ├── zba-extension.yml / _semantics.yml    # 地址生成扩展
    ├── zbb-extension.yml / _semantics.yml    # 基础位操作扩展
    ├── zbc.yml / _semantics.yml              #  carry-less multiply
    ├── zbs-extension.yml / _semantics.yml    # 单比特操作扩展
    ├── zacas.yml / _semantics.yml            #  CAS 原子操作
    ├── zabha.yml / _semantics.yml            #  Byte/Halfword 原子
    ├── zawrs.yml / _semantics.yml            #  Wait-on-Reservation-Set
    ├── zicbom/zicbop/zicboz.yml              #  Cache 管理
    ├── zicond.yml / _semantics.yml           #  条件零化
    ├── zicsr.yml / _semantics.yml            #  CSR 操作
    ├── zimop.yml / _semantics.yml            #  May-Be-Operations
    ├── zkn.yml / _semantics.yml              #  NIST 加密扩展
    ├── zks.yml / _semantics.yml              #  国密扩展
    ├── zfh.yml / zfhmin.yml / _semantics.yml #  Half-precision FP
    ├── fd-extension.yml / _semantics.yml     #  F/D 浮点扩展
    ├── c-extension.yml / _semantics.yml      #  压缩指令扩展 (RVC)
    └── v-extension*.yml                      #  向量扩展 (RVV)
```

## 文件命名约定

| 后缀/模式 | 含义 |
|-----------|------|
| `_meta.yml` | 架构级元信息，非指令列表 |
| `_formats.yml` | 指令编码格式定义 |
| `_semantics.yml` | 对应同名文件的语义补充 |
| `missing_*.yml` | LoongArch 中容易被忽略的基础指令别名 |
| `*-extension.yml` | RISC-V 标准扩展模块 |
| `z*-extension.yml` | RISC-V 子扩展（Z 扩展） |

## 查阅指南（按场景）

### 场景1：移植内联汇编 / 手写汇编

**问题**：某段 RISC-V 汇编需要改写成 LoongArch 汇编。

**步骤**：
1. 在 `riscv64/` 中找到对应指令（用 grep 搜索指令名）
2. 查看其 `encoding.match` 和 `assembly` 格式，确认操作数
3. 在 `ARCH_DIFF.md` 中查找是否有直接映射关系
4. 若无直接映射，在 `loongarch64/` 中搜索语义等价的指令（如 `semantics.operations[].op` 字段）
5. 注意 `loongarch64/_formats.yml` 中 LA64 的操作数命名（`$d`=`rd`, `$j`=`rs1`, `$k`=`rs2`）

### 场景2：处理原子操作移植

**问题**：`__atomic_compare_exchange_n` 或自旋锁相关代码。

**关键文件**：
- `loongarch64/atomics.yml` + `atomics_semantics.yml`
- `riscv64/a-extension.yml` + `zacas.yml`

**注意差异**：
- LA64 有独立的 `LL.W` / `SC.W`（带偏移量 `si14`）和 `AM*` 系列
- RV64 的 `LR.W` / `SC.W` 无偏移量，需单独计算地址
- LA64 的 `_DB` 后缀表示 Acquire/SeqCst 序，RV64 通过 `aq`/`rl` 位控制

### 场景3：SIMD/向量代码移植

**问题**：SSE/AVX/NEON/SVE 代码需要映射到 LA64 的 LSX/LASX 或 RV64 的 RVV。

**关键文件**：
- `loongarch64/lsx.yml`, `lasx.yml`
- `riscv64/v-extension*.yml`

**注意**：
- LSX = 128-bit 固定宽度（类似 SSE/NEON）
- LASX = 256-bit 固定宽度（类似 AVX）
- RVV = 可变向量长度（VLA），需额外处理 `vl` 设置

### 场景4：位操作优化（crypto、hash、compression）

**问题**：需要 `CLZ`、`CTZ`、`REV8`、`POPCNT` 等指令。

**关键文件**：
- `loongarch64/bitops.yml`（LA64 位操作在 base 中，非常完整）
- `riscv64/zbb-extension.yml`（RV64 位操作需 Zbb 扩展）

**映射速查**：
- `CLZ` / `CTZ` → LA64 有 `.W` 和 `.D` 版本，RV64 需 Zbb
- `REV8` → LA64 `REVB.D`，RV64 Zbb `rev8`
- `ANDN` / `ORN` / `XNOR` → RV64 Zbb 有，LA64 无直接对应，需用 `AND`+`NOR` 组合

### 场景5：浮点移植与 ABI 对齐

**问题**：浮点参数传递、FCSR 行为、NaN 处理。

**关键文件**：
- `loongarch64/_meta.yml` → `csrs.fcsr` / `frm` / `fflags`
- `riscv64/_meta.yml` → `csrs.fcsr` / `frm` / `fflags`
- `loongarch64/fp-common.yml`, `fp-s.yml`, `fp-d.yml`
- `riscv64/fd-extension.yml`, `zfh*.yml`

**差异**：
- LA64 有 `fcc` 条件标志寄存器（8个1位），RV64 无独立 FCC，条件通过整数寄存器传递
- LA64 FCSR 地址 `0x1`，RV64 FCSR 地址 `0x003`
- RV64 有 Zfh（半精度），LA64 需通过其他方式模拟

### 场景6：理解内存模型与屏障

**问题**：多线程代码中的 `memory_barrier`、`smp_mb()` 等。

**关键文件**：
- `loongarch64/_meta.yml` → `memory_model_details`
- `loongarch64/atomics.yml` → 各种 `_DB` 后缀指令

**核心差异**：
- LA64 默认 **Release Consistency**（弱于 TSO），显式使用 `DBAR`/`IBAR`
- RV64 默认 **TSO**（通过扩展可选弱序），使用 `FENCE`
- LA64 原子指令通过 `_DB` 后缀（`AMSWAP_DB.W` 等）控制序，RV64 通过指令内的 `aq`/`rl` 位

## YAML 字段速查

```yaml
instructions:
  - name: "指令助记符"
    group: "所属分组"
    encoding:
      match: "32位编码匹配模板（?表示可变位）"
    assembly: "汇编语法模板"
    format: "引用 _formats.yml 中的格式名"
    semantics:
      description: "人类可读描述"
      operations:
        - type: "操作类型（alu/load/store/atomic/branch/compare/cast/move/bit_field/select/read_time/trap）"
          op: "具体操作（Add/Sub/And/Or/Xor/Shl/Shr/Sar/Cas/Swap/Max/Min/Clz/Ctz/Rev8...）"
          dest: "目的操作数"
          src1/src2/addr/cond: "源操作数"
          size: "访问大小（字节）"
          signed: "是否符号扩展"
          ordering: "内存序（Relaxed/Acquire/Release/SeqCst）"
      implicit_pc_read: "是否隐式读 PC"
      implicit_pc_write: "是否隐式写 PC"
```

## 与外部工具配合使用

这些 YAML 文件可用于：
- **代码生成**：生成汇编器、反汇编器、模拟器的骨架代码
- **差异分析**：对比两个架构在特定功能域的指令覆盖度
- **翻译验证**：检查一条 LA64 指令的语义是否与 RV64 指令序列等价
- **测试生成**：根据语义描述生成随机测试用例
