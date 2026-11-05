# PHP 8.5.7 LoongArch64 JIT — 踩坑笔记

## 1. DynASM 构建流水线与直接缓冲发射不兼容

**坑：** DynASM 将 `|| LA64_RRI12(...)` 转换为 C 语句 `LA64_RRI12(...)`，
在 LA64 后端中这些是空语句（编译为无效果表达式），指令从未实际发射。

**表现：** JIT 编译"成功"但生成的代码缓冲区为空 → SIGILL 或无法启用 JIT。

**解决：** 所有 DynASM 指令发射必须通过 `la64_emit(ctx, ...)` 包裹。

## 2. LA64 RRI12 偏移是字节偏移，不是缩放偏移

**坑：** 从 aarch64 移植代码时，`offset / 8`、`offset / 4`、`offset / 2` 随处可见。
但 LA64 的 12 位立即数内存指令使用**原始字节偏移**。

```c
// aarch64 正确（缩放）
|| A64_STR32(reg, base, offset / 4);

// LA64 错误（缩放）
LA64_RRI12(LA64_ST_D, reg, base, offset / 8);  // ← 错了！

// LA64 正确（字节）
LA64_RRI12(LA64_ST_D, reg, base, offset);
```

## 3. `IR_REG_SPILLED(IR_REG_NONE)` 是 truthy 的

**坑：** `IR_REG_NONE` 定义为 `-1`。`IR_REG_SPILLED(r)` 对 `r == IR_REG_NONE`
返回 `-1`（非零 → truthy）。所以在检查 `IR_REG_SPILLED` 之前必须先确认
寄存器不是 `IR_REG_NONE`。

```c
// 错误 —— IR_REG_NONE 时也会进入
if (IR_REG_SPILLED(ctx->regs[i][0])) { ... }

// 正确 —— 先过滤 IR_REG_NONE
if (ctx->regs[i][0] != IR_REG_NONE && IR_REG_SPILLED(ctx->regs[i][0])) { ... }
```

## 4. 操作数编号：`ctx->regs[i][2]` 不一定是第二个操作数

**坑：** 复制 aarch64 的 `ir_emit_code` 调度代码时，以为 RETURN 的值在操作数 2，
但 `IR_RETURN` 指令的源值在**操作数 1**（`ctx->regs[i][1]`）。

`ctx->regs[i][k]` 中 k 的编号规则：
- `[0]`: def (结果)
- `[1]`: op1 (第一个源操作数)
- `[2]`: op2 (第二个源操作数)
- `[3]`: op3 (第三个源操作数，用于 STORE 等)

## 5. `ir_match_insn` 不可返回 0

**坑：** LA64 的 `ir_emit_code` 中 `default: return 0;` 导致规则为 0 的指令
在 `ir_coalesce` 中崩溃（`ir_vregs_overlap` 访问空 `live_intervals`）。

**解决：** `default:` 返回 `insn->op`（原始操作码值），确保框架有有效的规则值。

## 6. `ir_allocate_unique_spill_slots` 不可省略

**坑：** LA64 的 `ir_emit_code` 中注释掉了 `ir_allocate_unique_spill_slots`，
导致 spill 槽的 `stack_spill_pos` 未被正确分配 → `ir_vreg_spill_slot` 断言失败。

**解决：** 恢复 `ir_allocate_unique_spill_slots(ctx);` 调用。

## 7. CONFIGURE 路径问题

**坑：** 源码从 X86-64 机器复制到 LA64 机器后，`config.nice` 中配置路径
（`--prefix=/home/anuser/...`）是旧机器的。`make` 时因为旧路径上的文件
不存在而失败。

**解决：** 在新机器上重新运行 `./configure`。

## 8. `zend_jit_stub_handlers` 未分配

**坑：** `#if defined(_WIN32) || defined(IR_TARGET_AARCH64) || defined(IR_TARGET_LOONGARCH64)`
声明了 stub_handlers 为 `static void** ... = NULL;`，但实际分配代码只在
`_WIN32` 和 `IR_TARGET_AARCH64` 下运行。LA64 遇到 NULL 指针写入 → SEGV。

**解决：** 在所有条件编译处添加 `|| defined(IR_TARGET_LOONGARCH64)`。

## 9. `ir_get_target_constraints` 中 `insn` 变量作用域

**坑：** `insn` 在函数开始时声明但未初始化，只在部分 case 分支中赋值。
`default:` 分支直接访问 `insn->op`，此时 `insn` 可能是垃圾指针。

**解决：** 在函数开始处初始化：`const ir_insn *insn = &ctx->ir_base[ref];`。

## 10. 函数 JIT vs 追踪 JIT 差异

**坑：** tracing JIT 工作正常但 function JIT 崩溃。两种 JIT 模式的编译路径
使用了相同的 `ir_emit_code` 函数，但函数 JIT 会编译完整的函数体，涉及
更多指令类型（CALL、复杂的条件分支、异常处理）。

**已知缺失：**
- `IR_CALL` → 发射 `NOP` 占位符（未实现函数调用）
- `IR_IF_INT` → 条件分支偏移计算不完整
- `IR_END/IR_LOOP_END` → 基本块尾分支未实现

## 11. LA64 栈偏移编码

`ADDI_D sp, sp, offset` 的 12 位立即数是**有符号**的。
负偏移需要正确处理符号扩展：
```c
// 正确方式
uint32_t encoded_offset = (uint32_t)((int32_t)offset & 0xfff);
la64_emit(ctx, LA64_RRI12(LA64_ADDI_D, IR_REG_SP, IR_REG_SP, encoded_offset));
```

## 12. 条件分支偏移

LA64 的 BEQZ/BNEZ 指令使用**以指令字计数的偏移**（21 位有符号），
不是字节偏移。生成 branches 时需要计算目标地址与当前地址的差值
（以指令为单位）。

当前 `ir_emit_code` 的 IF_INT 处理器尚未实现正确的分支偏移计算。
