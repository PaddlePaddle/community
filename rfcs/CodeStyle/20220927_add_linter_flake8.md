# 代码风格检查工具 Flake8 引入计划

| 任务名称     | 代码风格检查工具 Flake8 引入计划 |
| ------------ | -------------------------------- |
| 提交作者     | Nyakku Shigure(@SigureMo)        |
| 提交时间     | 2022-09-27                       |
| 版本号       | v0.1                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20220927_add_linter_flake8.md    |

## 一、概述

### 1、相关背景

Linter 是代码开发过程中的一个重要工具，它可以帮助开发者在编写代码时就发现一些格式、语法甚至一些逻辑错误。由于 Python 是一门解释型语言，并没有编译时来对一些错误进行检查，很多问题可能留到运行时才能暴露出来，因此 Linter 将这些问题提前到开发时，使得开发者能够及时发现并修复问题，提高代码质量。

Paddle 目前在 C++ 端代码使用 cpplint 作为 Linter，Python 端却并没有一个 Linter 来规范代码（虽然目前有使用 pylint 但仅仅用于检查 docstring）。

Flake8 是一个被广泛使用的 Python Linter，它利用插件系统集成了多个 Linter 工具，默认的插件包含了 pycodestyle、pyflakes、mccabe 三个工具，分别用于检查代码风格、语法问题和循环复杂度。此外 Flake8 还支持安装第三方插件，以对代码进行更全面的检查。

本任务来源于 [Call for contribution - Flake8](../../pfcc/call-for-contributions/code_style_flake8.md)

### 2、功能目标

修复 Paddle 当前代码中 Flake8 存量问题，并在 pre-commit 和 CI 中增加 Flake8 检查项。

### 3、意义

规范代码风格，提高代码质量，并能够在开发时发现一些潜在的逻辑问题。

## 二、实现方案

### 阻止增量问题

1. 添加 Flake8 的配置文件 `.flake8`，在 pre-commit 中添加 flake8 hook（由 pre-commit clone repo）
2. 将 flake8 修改为 local hook，添加相应的 hook 脚本，并在 Docker 镜像中提前安装 flake8
3. 更新代码风格检查指南文档，添加 flake8

### 修复存量问题

Flake8 默认启用的三个工具（pycodestyle、pyflakes、mccabe）共包含 132 个错误码，在 Flake8 引入任务启动前 Paddle 共存在 64 个错误码。

Flake8 的 132 个错误码中有 11 个由于未被广泛认可而默认未启用的，此外还有一些错误码（如 E203）还被认为过分严苛也没有被社区广泛认可，这些错误码可以在 Flake8 配置文件中排除掉，不作考虑。

对于其余错误码，先在 Flake8 配置文件中排除掉，之后每修复一个错误码的存量问题后在配置中移除该错误码，使 pre-commit 能够阻止该错误码增量。

TODO：扫描统计

TODO：F401 解决方案

## 三、任务分工和排期规划

由 Flake8 小组自行认领任务，每人负责部分错误码的部分目录。由于任务尚未开始，具体排期尚无法确定。

## 四、其他注意事项及风险评估

TODO

## 五、影响面

TODO

## 名词解释

- Linter：代码检查工具（含代码风格、语法、逻辑等等）
- Formatter：代码格式化工具

## 附件及参考资料

- [Flake8 Error Code](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- [pycodestyle Error Code](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes)
