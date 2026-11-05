# PHP 8.5.7 LoongArch64 JIT — 移植补丁说明

## 补丁系列总览

```
Patch 0001 │ 49KB │ ir_match_insn/ir_get_target_constraints/规则枚举
   └─ 修复 JIT IR 匹配框架的核心崩溃
   
Patch 0002 │  4KB │ 稳定的构建配置 + 文档
   └─ 构建依赖和配置修正
   
Patch 0003 │  2KB │ zend_jit.c stub_handlers 分配
   └─ 修复 LA64 下 NULL 指针写入
   
Patch 0004 │ 45KB │ dasc 源文件 + ir_emit.c/ir_ra.c 框架补丁
   └─ JIT 后端核心、框架断言补丁
```

## 各补丁详解

### Patch 0001: JIT IR 匹配框架修复

**解决的问题：**
- `ir_match_insn()` LA64 存根返回 0，导致所有指令规则为空
- `ir_get_target_constraints()` 中 `insn` 指针在 `default:` 分支未初始化 → SEGV
- `IR_RULES` 枚举只有 `STUB_OP`，缺少必要规则定义

**修改的文件：**
- `ext/opcache/jit/ir/ir_loongarch64.dasc` — 新增 `ir_match_insn()` 完整实现
- `ext/opcache/jit/ir/ir_emit_loongarch64.h` — 规则枚举生成物

### Patch 0002: 稳定构建

**解决的问题：**
- 构建系统配置修正
- 状态文档添加

### Patch 0003: stub_handlers 分配

**解决的问题：**
- `zend_jit_stub_handlers` 只在 `_WIN32` 和 `AARCH64` 条件分配
- LA64 被排除导致 NULL 指针写入崩溃

**修改的文件：**
- `ext/opcache/jit/zend_jit.c` — 添加 `defined(IR_TARGET_LOONGARCH64)`

### Patch 0004: dasc 源文件 + 框架补丁

**解决的问题：**
- `ir_emit_loongarch64.dasc` 的完整 LA64 后端代码
- `ir_ra.c` 断言补丁（`IR_ASSERT` 改为条件跳过）

**修改的文件：**
- `ext/opcache/jit/ir/ir_loongarch64.dasc` — JIT 后端核心
- `ext/opcache/jit/ir/ir_emit.c` — `ir_get_call_conv_dsc` LA64 路径
- `ext/opcache/jit/ir/ir_ra.c` — 断言兼容性补丁

## 后端后处理说明

`ir_emit_loongarch64.h` 由 DynASM 从 `.dasc` 文件生成，必须进行以下后处理：

### 1. 指令宏包裹
将 DynASM 输出的裸指令宏包裹为 `la64_emit()`：
```c
// 修复前（编译为无操作）
LA64_RRI12(LA64_ADD_D, r4, r4, 0);

// 修复后（实际写入指令）
la64_emit(ctx, LA64_RRI12(LA64_ADD_D, r4, r4, 0));
```

### 2. 偏移编码修正
LA64 RRI12 内存偏移为**原始字节偏移**，不可按访问大小缩放：
```c
// 错误（aarch64 习惯）
LA64_RRI12(LA64_ST_D, reg, base, offset / 8);

// 正确（LA64 使用字节偏移）
LA64_RRI12(LA64_ST_D, reg, base, offset);
```

### 3. 操作数编号修正
`ctx->regs[i][2]` → `ctx->regs[i][1]`（操作数 2 → 操作数 1）：
- IR_RETURN_INT/FP: 返回值在操作数 1，不是 2
- 所有 `insn->op2` 使用处需确认是否应为 `op1`

### 4. IR_REG_NONE 守卫
`IR_REG_SPILLED(IR_REG_NONE)` 是 truthy 的！
```c
// 错误
if (IR_REG_SPILLED(ctx->regs[i][0])) { ... }

// 正确
if (ctx->regs[i][0] != IR_REG_NONE && IR_REG_SPILLED(ctx->regs[i][0])) { ... }
```

## 移植策略

1. **优先 tracing JIT**：tracing JIT 走热路径，涉及的指令类型较少
2. **function JIT 待修复**：需要实现完整的 IR_CALL 函数调用和条件分支偏移计算
3. **编码优先使用 #define**：所有 LA64 指令编码使用宏（`LA64_ADD_D` 等），不硬编码
