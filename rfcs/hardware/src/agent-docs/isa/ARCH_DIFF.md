# LoongArch64 vs RISC-V64 架构差异速查

> 本文档摘录两大架构在指令集模型中的核心差异，供 Agent 在移植、优化、调试时快速查阅。
> 完整数据请参阅各自的 YAML 文件。

---

## 1. 寄存器模型

| 特性 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 通用寄存器 | `r0`~`r31` | `x0`~`x31` |
| 零寄存器 | `r0` = `$zero` | `x0` = `zero` |
| 栈指针 | `r3` = `$sp` | `x2` = `sp` |
| 返回地址 | `r1` = `$ra` | `x1` = `ra` |
| 线程指针 | `r2` = `$tp` | `x4` = `tp` |
| 浮点寄存器 | `f0`~`f31` | `f0`~`f31` 或 `ft0`~`ft11`/`fs0`~`fs11`/`fa0`~`fa7` |
| 浮点条件码 | **有独立 `fcc0`~`fcc7`**（1位） | 无，通过整数比较实现 |
| 向量/SIMD | `v0`~`v31` (128b LSX) / `x0`~`x31` (256b LASX) | `v0`~`v31` (RVV, VLA) |
| 浮点/向量共享 | LA64 的 `f`/`v`/`x` 共享存储（`vector_shared`） | RV64 浮点与向量独立 |
| PC | 隐式，非编程寄存器 | 隐式，非编程寄存器 |

### 寄存器命名对照表（函数调用约定）

| 作用 | LA64 | RV64 |
|------|------|------|
| 返回值 | `$a0`=`r4`, `$a1`=`r5` | `a0`=`x10`, `a1`=`x11` |
| 参数1~8 | `$a0`~`$a7` = `r4`~`r11` | `a0`~`a7` = `x10`~`x17` |
| 保存寄存器 | `$s0`~`$s8` = `r23`~`r31` | `s0`~`s11` = `x8`,`x9`,`x18`~`x27` |
| 临时寄存器 | `$t0`~`$t8` = `r12`~`r20` | `t0`~`t6` = `x5`~`x7`,`x28`~`x31` |
| 全局指针 | 无对应 | `gp`=`x3` |
| 框指针 | `$fp`=`r22` | `s0`/`fp`=`x8` |

### ABI 差异

| 特性 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 系统调用号 | `$a7` = `r11` | `a7` = `x17` |
| 系统调用返回 | `$a0` = `r4` | `a0` = `x10` |
| 退出系统调用 | 93, 94 | 93, 94 |
| 栈指针 | `$sp` = `r3` | `sp` = `x2` |
| 返回地址 | `$ra` = `r1` | `ra` = `x1` |

---

## 2. 指令格式

| 特性 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 指令长度 | 固定 32-bit | 固定 32-bit + 可选16-bit RVC |
| 格式数量 | ~26种格式（DJK/DJ/DI20/I26 等） | ~20种格式（R/I/S/B/U/J + C + V 等） |
| 编码特点 | 紧凑型，操作码占比高 | 可扩展型，opcode 分布在低位 |
| 立即数范围 | 多种宽度（4/5/6/7/8/10/12/14/16/20/26-bit） | I-type 12-bit，U-type 20-bit upper，J-type 20-bit |

### LA64 主要指令格式

| 格式 | 含义 | 类似 RV64 |
|------|------|------------|
| `DJK` | rd, rj, rk | R-type |
| `DJ` | rd, rj | I-type-unary / R-type-unary |
| `DJI12` | rd, rj, simm12 | I-type |
| `DI20` | rd, imm20 | U-type/LUI 类似 |
| `I26` | offset26 | J-type 类似 |
| `JI21` | rj, offset21 | 无直接对应 |
| `DJKCa` | rd, rj, rk, ca | 无（含 fcc 条件） |

### RV64 主要指令格式

| 格式 | 含义 | 类似 LA64 |
|------|------|------------|
| `R-type` | rd, rs1, rs2 | DJK |
| `I-type` | rd, rs1, imm | DJI12 |
| `U-type` | rd, imm[31:12] | DI20 类似 |
| `J-type` | rd, offset | I26 类似 |
| `S-type` | rs2, offset(rs1) | 存储指令 |
| `B-type` | rs1, rs2, offset | 分支指令 |
| `C-*` | 16-bit 压缩 | 无（LA64 无 RVC 类似物） |
| `V-*` | 向量指令 | LSX/LASX 类似 |

---

## 3. 内存模型与序一致性

| 特性 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 默认内存模型 | **Release Consistency** | **TSO** （可通过 Ztso 选择弱序） |
| 屏障指令 | `DBAR`（数据）, `IBAR`（指令） | `FENCE`（细粒度控制） |
| 原子序控制 | 指令后缀 `_DB` 切换 | 指令内 `aq`/`rl` 位 |
| 强度对比 | 弱于 TSO，更接近 ARM/POWER | 类似 x86 TSO（默认） |

### 原子操作对照

| 操作 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| Load-Linked | `LL.W` / `LL.D` （带 `si14` 偏移） | `LR.W` / `LR.D` （无偏移，地址在 rs1） |
| Store-Conditional | `SC.W` / `SC.D` （带 `si14` 偏移） | `SC.W` / `SC.D` （无偏移） |
| 原子交换 | `AMSWAP.W` / `AMSWAP.D` | `AMOSWAP.W` / `AMOSWAP.D` |
| 原子加法 | `AMADD.W` / `AMADD.D` | `AMOADD.W` / `AMOADD.D` |
| 原子比较交换 | `AMCAS.W` / `AMCAS.D` | 需 Zacas 扩展 `AMOCAS.W/D` |
| 最大值 | `AMMAX.W` / `AMMAX.D` | `AMOMAX.W` / `AMOMAX.D` |
| 无符号最大 | 无直接对应 | `AMOMAXU.W` / `AMOMAXU.D` |
| 序强度 | `_DB` = Acquire/SeqCst | `aq`/`rl` 位 |

**移植注意**：
- LA64 的 `LL`/`SC` 带有 `si14` 偏移，直接写入指令；RV64 需先计算地址到 `rs1`
- LA64 原子指令的序控制通过指令变种（`AMSWAP_DB.W` vs `AMSWAP.W`），不是通用位字段

---

## 4. 位操作与位字段指令

### LA64 独有（在 `bitops.yml` 中，无需扩展）

| 指令 | 功能 | RV64 对应 |
|------|------|-----------|
| `CLO.W` / `CLO.D` | 计数前导1 | 无直接对应（Zbb 只有 `CLZ`） |
| `CTO.W` / `CTO.D` | 计数尾随1 | 无直接对应（Zbb 只有 `CTZ`） |
| `REVB.2H` / `.4H` / `.2W` / `.D` | 字节反序 | `REV8` (Zbb) |
| `REVH.2W` / `.D` | 半字反序 | 无直接对应 |
| `REVBIT.4B` / `.8B` / `.W` / `.D` | 位反序 | 无直接对应 |
| `BYTEPICK.W` | 字节择选 | 无直接对应 |
| `BSTRINS.W` / `.D` | 位字段插入 | 无（需多条指令） |
| `BSTRPICK.W` / `.D` | 位字段提取 | 无（需多条指令） |
| `CRC.W.B.W` 等 | CRC32/CRC32C | 无（需软件实现） |
| `SLADD.W` / `SLADD.D` | 左移加法 | `SH1ADD` 等 (Zba) |

### RV64 独有（需对应扩展）

| 指令 | 扩展 | LA64 对应 |
|------|-------|-----------|
| `ANDN` | Zbb | `AND` + `NOR` 组合 |
| `ORN` | Zbb | `OR` + `NOR` 组合 |
| `XNOR` | Zbb | `XOR` + `NOR` 组合 |
| `CPOP` / `CPOPW` | Zbb | 无直接对应（软件） |
| `MAX` / `MAXU` / `MIN` / `MINU` | Zbb | 无直接对应（软件） |
| `ORC.B` | Zbb | 无直接对应 |
| `ROL` / `ROLW` / `ROR` / `RORW` | Zbb | `ROTR` + 调换操作数 |
| `RORI` / `RORIW` | Zbb | `ROTRI` |
| `PACK` | Zbb | 无直接对应 |
| `SH1ADD` / `SH2ADD` / `SH3ADD` | Zba | `SLADD` 类似 |
| `CLMUL` / `CLMULH` / `CLMULR` | Zbc | 无直接对应 |
| `BCLR` / `BEXT` / `BINV` / `BSET` | Zbs | `BSTRPICK`/`BSTRINS` 类似 |
| `C.ZERO.eqz` / `C.ZERO.nez` | Zicond | `MASKEQZ` / `MASKNEZ` |

### 共通位操作

| 功能 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 前导零计数 | `CLZ.W` / `CLZ.D` | `CLZ` / `CLZW` (Zbb) |
| 后导零计数 | `CTZ.W` / `CTZ.D` | `CTZ` / `CTZW` (Zbb) |
| 字节反序 | `REVB.D` | `REV8` (Zbb) |
| 符号扩展 | `SEXT.B` / `SEXT.H` / `SEXT.W` | `SEXT.B` / `SEXT.H` (Zbb) |
| 左移加 | `SLADD.W` | `SH1ADD` / `SH2ADD` / `SH3ADD` (Zba) |

---

## 5. 浮点与向量

| 特性 | LoongArch64 | RISC-V64 |
|------|-------------|----------|
| 浮点精度 | 单精度 `fp-s`，双精度 `fp-d` | F扩展单精度，D扩展双精度 |
| 半精度 | 无原生支持 | `Zfh` / `Zfhmin` 扩展 |
| 浮点条件码 | `fcc0`~`fcc7`（1位，独立寄存器） | 通过整数比较指令实现 |
| FCSR 地址 | `0x1` | `0x003` |
| 向量扩展 | LSX (128b) + LASX (256b)，固定宽度 | RVV (VLA, vlenb 可变) |
| 向量寄存器共享 | `f`/`v`/`x` 共享存储（LASX 覆盖 f/v） | 浮点与向量独立 |
| 浮点操作类型 | 有条件移动 `MOVCF2GR` 等 | 无条件码移动，需借助整数指令 |

---

## 6. 特殊扩展与独特指令

### LoongArch LBT 扩展（二进制翻译辅助）

在 `_meta.yml` 中定义了专门用于模拟 x86 EFLAGS 的指令：

| 指令 | 功能 |
|------|------|
| `X86MUL_B`/`H`/`W`/`D` | 模拟 x86 乘法标志 |
| `X86DIV_B`/`H`/`W`/`D` | 模拟 x86 除法 |
| `X86BT` | 模拟 x86 位测试 |
| `X86MOV_E` | 模拟 x86 条件移动/设置 |
| `X86FLG` | 读写模拟的 EFLAGS |

> **Agent 注意**：在移植涉及二进制翻译工具（如 QEMU User Mode）时，需特别关注 LBT 扩展的存在。普通应用软件移植不会直接使用这些指令。

### RISC-V 压缩指令 (RVC)

| 特性 | RISC-V64 | LoongArch64 |
|------|----------|-------------|
| 16-bit 指令 | 有（`C` 扩展） | 无（所有指令均为 32-bit） |
| 寄存器子集 | C 扩展使用 `x8`~`x15` | N/A |
| 影响 | 反汇编/调试时需处理 16-bit 指令 | 反汇编简化 |

---

## 7. 快速映射表（常见内联汇编情景）

| 功能 | LoongArch64 | RISC-V64 | 备注 |
|------|-------------|----------|------|
| 64-bit 加法 | `ADD.D` | `ADD` | |
| 32-bit 加法（符号扩展） | `ADD.W` | `ADDW` | |
| 立即加 | `ADDI.D` | `ADDI` | LA64 I12, RV64 I-type 12 |
| 立即加32-bit | `ADDI.W` | `ADDIW` | |
| 算术右移 | `SRA.D` / `SRAI.D` | `SRA` / `SRAI` | |
| 逻辑右移 | `SRL.D` / `SRLI.D` | `SRL` / `SRLI` | |
| 左移 | `SLL.D` / `SLLI.D` | `SLL` / `SLLI` | |
| 载入字 | `LD.W` | `LW` | |
| 载入双字 | `LD.D` | `LD` | |
| 存储字 | `ST.W` | `SW` | |
| 存储双字 | `ST.D` | `SD` | |
| 跳转 | `B` | `J` | LA64 I26, RV64 J-type |
| 条件跳转（等于） | `BEQ` | `BEQ` | |
| 条件跳转（小于） | `BLT` | `BLT` | |
| 无条件跳转 | `BL` | `JAL` | LA64 `BL` 隐含 `$ra`，RV64 `JAL rd, offset` |
| 返回 | `JIRL` | `JALR` | LA64 `JIRL rd, rj, 0` = RV64 `JALR rd, rs1, 0` |
| PC译址高位 | `PCADDU12I` | `AUIPC` | **LA64 的 PCADDU12I 不做 PC 页对齐**，AUIPC 也不做 |
| PC译址低位（载入） | `LD.D rd, rj, si12` | `LD rd, si12(rs1)` | LA64 的 `LD.D` 要求 8 字节对齐 |
| 左移加法 | `ALSL.D rd, rj, rk, sa` | `SH1ADD`等 | LA64: `rd = rk + (rj << sa)`，sa 是直接移位量 |
| 条件移动 | `MOVCF2GR` 等 | 需通过分支实现 | |
| 等于零则掩码 | `MASKEQZ` | `CZERO.EQZ` (Zicond) | |
| 不等于零则掩码 | `MASKNEZ` | `CZERO.NEZ` (Zicond) | |
| 符号扩展字节 | `SEXT.B` | `SEXT.B` (Zbb) | |
| 符号扩展半字 | `SEXT.H` | `SEXT.H` (Zbb) | |
| 零扩展半字 | `ZEXT.H` 无原生 | `ZEXT.H` (Zbb) | LA64 需 `ANDI` |
| 计数前导零 | `CLZ.D` | `CLZ` (Zbb) | |
| 计数尾随零 | `CTZ.D` | `CTZ` (Zbb) | |
| 字节反序 | `REVB.D` | `REV8` (Zbb) | |
| 旋转右 | `ROTR.D` / `ROTRI.D` | `ROR` / `RORI` (Zbb) | |
| 旋转左 | 无直接 | `ROL` / `ROLW` (Zbb) | LA64 `ROTR` 调换操作数模拟 |
| 分支链接 | `BL` | `CALL` 约定 | |
| 读时间戳 | `RDTIME.D` | `RDINSTRET` 类似 | |
| CPU 配置 | `CPUCFG` | 无直接对应 | |
| 堆栈分配 | `ADDI.D $sp, $sp, -imm` | `ADDI sp, sp, -imm` | |

---

## 8. 移植常见陷阱

### 陷阱1：条件码传递

- **LA64**: 浮点比较结果放入 `fcc`。`BCEQZ`/`BCNEZ` 直接判断 `fcc`
- **RV64**: 浮点比较通常通过 `FEQ.S`/`FLT.S`/`FLE.S` 将结果写入整数寄存器，再用 `BEQ`/`BNE` 判断
- **移植时**: 不要尝试直接映射 LA64 的 `fcc` 相关汇编到 RV64，需重构逻辑

### 陷阱2：边界检查

- **LA64**: 有原生边界检查指令 `BCEQZ`/`BCNEZ` 和 `BOUND`系列
- **RV64**: 无原生边界检查，需软件实现
- **移植时**: 当看到 `bound.yml` 中的指令时，需将其展开为多条比较/分支指令

### 陷阱3：立即数构建

- **LA64**: 使用多条指令构建 64-bit 立即数：`LU12I.W` + `ORI` + `LU32I.D` + `LU52I.D`
- **RV64**: `LUI` + `ADDI` 构建低 32-bit，高 32-bit 需 `LUI` + `SLLI` + `ADDI`
- **移植时**: LA64 的 `LU32I.D` 和 `LU52I.D` 是特有的，不要简单映射为 RV64 的 `LUI`

### 陷阱4：内存序一致性

- **LA64**: 原子指令的 `DB` 后缀表示强序，但非所有组合都是 SeqCst
- **RV64**: `FENCE` 和 `aq`/`rl` 位提供精细控制
- **移植时**: 不要假设两者的原子序行为完全等价。LA64 的 Release Consistency 更弱，需更多显式屏障

### 陷阱5：SIMD 宽度错配

- **LA64 LSX** = 128-bit 固定宽度，类似 SSE/NEON
- **LA64 LASX** = 256-bit 固定宽度，类似 AVX
- **RV64 RVV** = 可变长度，`vsetvli` 设置 `vl`
- **移植时**: 不要假设 LA64 的向量长度与 RVV 相同。从 LSX 到 RVV 的移植需重新实现向量长度管理逻辑
