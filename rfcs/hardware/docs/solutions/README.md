# 移植方案索引

> 本目录存放各软件包在 LoongArch64 上的完整移植方案。每个子目录包含该软件包的
> 构建步骤、补丁说明、测试报告、spec 文件及相关源码。

## 方案清单

| 软件包 | 版本 | 类别 | 状态 | 关键补丁 |
|--------|------|:---:|:----:|----------|
| [Python 2.7](./python2.7/) | 2.7.18 | **B** | ✅ 完成 | config.guess 更新、OpenSSL 3.x 兼容 |
| [PHP JIT](./php-jit/) | 8.5.7 | **E** | ⚠️ tracing 可用 | LA64 JIT 后端 4-patch 系列 |

### 类别说明

| 类别 | 含义 | 预计工作量 |
|:---:|------|:---:|
| A | 架构无关 | 分钟 |
| B | 便携 C/C++ | 十分钟 |
| C | SIMD/Intrinsic | 小时 |
| D | 内联汇编 | 天 |
| E | JIT/动态编译 | 周 |

## Python 2.7.18 (类别 B)

- **移植要点**: config.guess 替换、`--with-system-ffi`（捆绑 libffi 过旧）、OpenSSL 3.x 版本宏适配
- **补丁数**: 2 个有效补丁 (+ 1 个废弃)
- **RPM 产出**: 6 个二进制包 + 1 个 SRPM
- **测试**: 核心模块 smoke test 全通过，完整套件等待回归
- **文档**: [BUILD.md](./python2.7/BUILD.md) · [PATCHES.md](./python2.7/PATCHES.md) · [TEST-REPORT.md](./python2.7/TEST-REPORT.md) · [NOTES.md](./python2.7/NOTES.md)

## PHP 8.5.7 JIT (类别 E)

- **移植要点**: LA64 JIT 后端完整实现（ir_match_insn, ir_emit_code, stub_handlers）、DynASM 后处理（指令包裹/偏移修正/操作数编号）、栈偏移编码
- **补丁数**: 4 个补丁 (总计 ~100KB)
- **JIT 模式**: tracing JIT 99.9% 测试通过，function JIT 有 SIGILL 待修复
- **性能**: tracing JIT 加速比 1.19x-1.54x（与无 JIT 对比）
- **文档**: [BUILD.md](./php-jit/BUILD.md) · [PATCHES.md](./php-jit/PATCHES.md) · [TEST-REPORT.md](./php-jit/TEST-REPORT.md) · [NOTES.md](./php-jit/NOTES.md)（12 个踩坑记录）

## 目录结构约定

```
solutions/<软件名>/
├── BUILD.md           # 完整构建步骤（含双架构对比）
├── PATCHES.md         # 补丁说明（每处修改的原因和影响）
├── TEST-REPORT.md     # 测试对比报告
├── NOTES.md           # 踩坑笔记 / 杂项记录
├── SOURCES/           # 补丁文件 (*.patch)
├── SPECS/             # RPM spec 文件
├── SRPMS/             # 源码 RPM 包
└── RPMS/              # 二进制 RPM 包
    └── loongarch64/
```

---

> **维护记录**
>
> | 日期 | 更新 | 更新者 |
> |------|------|--------|
> | 2026-06-09 | 初始化：Python 2.7.18 移植方案 | — |
> | 2026-06-09 | 新增：PHP 8.5.7 JIT 移植方案 | — |
> | 2026-06-09 | 整理：统一目录结构，创建索引 | Hermes Agent |
