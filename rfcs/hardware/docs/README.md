# 移植方案与经验记录

> 本目录存放 OpenClaw 移植工作中积累的**具体方案、构建记录、修复笔记**。
> 与 `src/agent-docs/` 的区别：那里是 Agent 行为规范和技术参考；这里是实际移植过程中产生的事实性记录。

## 核心文档

| 文档 | 说明 |
|------|------|
| [自动化移植方案（初版）](./loongarch-automated-porting-proposal.md) | **总体方案**：架构设计、5级分类体系、标准流水线、质量体系、实施路线图 |
| [使用手册](./loongarch-porting-user-manual.md) | **操作指南**：环境搭建、4类场景操作、Agent 协作、故障排查 |
| [测试报告](./reports/loongarch-porting-test-report.md) | **验证报告**：Python 2.7.18 / PHP 8.5.7 JIT 测试结果 |
| [移植方案索引](./solutions/README.md) | 已完成的软件包移植方案清单 |

## 目录结构

```
docs/
├── README.md                                  # 本文件
├── loongarch-automated-porting-proposal.md     # 自动化移植方案文档
├── loongarch-porting-user-manual.md            # 使用手册
├── solutions/                                  # 各软件包移植方案
│   ├── README.md                               # 方案索引
│   ├── python2.7/                              # ★ Python 2.7.18 (B类, 完成)
│   │   ├── BUILD.md / PATCHES.md / TEST-REPORT.md / NOTES.md
│   │   ├── SOURCES/ (3 补丁) / SPECS/ / SRPMS/ / RPMS/
│   └── php-jit/                                # ★ PHP 8.5.7 JIT (E类, tracing可用)
│       ├── BUILD.md / PATCHES.md / TEST-REPORT.md / NOTES.md
│       ├── SOURCES/ (cover letter) / SPECS/
├── reports/                                    # 测试报告
│   └── loongarch-porting-test-report.md        # 自动化移植测试报告
└── build-logs/                                 # 构建日志归档 (待填充)
```

## 与 src/agent-docs 的分工

| 类型 | 位置 | 性质 |
|------|------|------|
| Agent 行为合约、ISA 模型、通用移植指南 | `src/agent-docs/` | 规范与参考，长期维护 |
| 具体项目的移植步骤、补丁、日志 | `docs/` | 事实记录，按需增减 |

## 阅读指南

| 我想... | 先读 |
|---------|------|
| 了解整体方案 | `loongarch-automated-porting-proposal.md` |
| 动手移植一个包 | `loongarch-porting-user-manual.md` |
| 查看测试结果 | `reports/loongarch-porting-test-report.md` |
| 参考已有案例 | `solutions/README.md` |

## 使用方式

- Agent 完成一次移植后，将方案整理到 `docs/solutions/<软件名>/`
- 关键构建日志可移入 `docs/build-logs/` 归档
- 移植完成报告放入 `docs/reports/`
