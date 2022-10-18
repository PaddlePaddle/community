# 代码风格检查工具 Flake8 引入计划

| 任务名称     | 代码风格检查工具 Flake8 引入计划 |
| ------------ | -------------------------------- |
| 提交作者     | Nyakku Shigure(@SigureMo)        |
| 提交时间     | 2022-09-27                       |
| 版本号       | v1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20220927_introducing_flake8.md   |

## 一、概述

### 1、相关背景

Linter 是代码开发过程中的一个重要工具，它可以帮助开发者在编写代码时就发现一些格式、语法甚至一些逻辑错误。由于 Python 是一门解释型语言，并没有编译时来对一些错误进行检查，很多问题可能留到运行时才能暴露出来，而 Linter 则可以将这些问题提前到开发时，使得开发者能够及时发现并修复问题，提高代码质量。

Paddle 目前在 C++ 端代码使用 cpplint 作为 Linter，Python 端却并没有一个 Linter 来规范代码（虽然目前有使用 pylint 但仅仅用于检查 docstring）。

Flake8 是一个被广泛使用的 Python Linter，它利用插件系统集成了多个 Linter 工具，默认的插件包含了 pycodestyle、pyflakes、mccabe 三个工具，分别用于检查代码风格、语法问题和循环复杂度。此外 Flake8 还支持安装第三方插件，以对代码进行更全面的检查。

本任务来源于 [Call for contribution - Flake8](../../pfcc/call-for-contributions/code_style_flake8.md)

### 2、功能目标

修复 Paddle 当前代码中 Flake8 存量问题，并在 pre-commit 和 CI 中增加 Flake8 检查项。

### 3、意义

规范代码风格，提高代码质量，并使开发者能够在开发时发现一些潜在的逻辑问题。

## 二、实现方案

### 阻止增量问题

1. 添加 Flake8 的配置文件 `.flake8`，在 pre-commit 中添加 flake8 hook（由 pre-commit clone repo）
2. 将 flake8 修改为 local hook，添加相应的 hook 脚本，并在 Docker 镜像中提前安装 flake8
3. 更新代码风格检查指南文档，添加 flake8

### 修复存量问题

Flake8 默认启用的三个工具（pycodestyle、pyflakes、mccabe）共包含 132 个错误码，在 Flake8 引入任务启动前 Paddle 共存在 64 个错误码。

Flake8 的 132 个错误码中有 11 个由于未被广泛认可而默认未启用的，此外还有一些错误码（如 E203）还被认为过分严苛也没有被社区广泛认可，这些错误码可以在 Flake8 配置文件中排除掉，不作考虑。

对于其余错误码，先在 Flake8 配置文件中排除掉，之后每修复一个错误码的存量问题后在配置中移除该错误码，使 pre-commit 能够阻止该错误码增量。

以下是在完成 trailing whitespace（W291、W293）相关修复后的统计

```text
Type: E (26468)
E101    11
E121    8
E122    81
E123    12
E125    168
E126    723
E127    140
E128    207
E129    9
E131    45
E201    29
E202    11
E203    32
E225    61
E226    93
E228    3
E231    60
E241    2
E251    109
E261    11
E262    238
E265    925
E266    116
E271    4
E272    1
E301    7
E302    3
E303    7
E305    2
E306    1
E401    19
E402    2666
E501    19252
E502    400
E701    108
E711    166
E712    340
E713    22
E714    4
E721    8
E722    149
E731    62
E741    153

Type: F (9895)
F401    6750
F402    1
F403    57
F405    556
F522    1
F524    1
F541    33
F601    7
F631    2
F632    18
F811    177
F821    88
F841    2204

Type: W (1414)
W191    11
W503    279
W504    949
W601    3
W605    172
```

> **Note**
>
> - E、W 错误码详情见：[pycodestyle Error Code](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes)
> - F 错误码详情见：[Flake8 Error Code](https://flake8.pycqa.org/en/latest/user/error-codes.html)

其中 trailing whitespace 相关问题（W291、W293）已经被 pre-commit hook [trailing-whitespace](https://github.com/pre-commit/pre-commit-hooks#trailing-whitespace) 解决，Tabs 相关问题（E101、W191）已经被 pre-commit hook [remove-tabs](https://github.com/Lucas-C/pre-commit-hooks#usage) 解决。

此外存在一个语法错误（E999）已经在之前的 [#45287](https://github.com/PaddlePaddle/Paddle/pull/45287) 解决。

black（Formatter）能自动解决大多数格式上的问题（E 错误码），[introducion black RFC](./20221018_introducing_black.md) 中详细阐述了使用 black 来替代 yapf 所带来的优势，并将在近期完成使用 black 替代 yapf 的工作。引入 black 可以解决 E121、E122 等大多数 E 错误码。

其余错误码需要根据情况来处理：

- 存量较少的错误码（低于 30）：手动修复即可
- 存量较大的错误码：
  - 可利用 autoflake 修复部分 F 错误码
  - 可利用 autopep8 修复部分 black 剩余的 E 错误码
  - 上述两个工具无法修复的且存量较大的错误码需要编写脚本修复或者手动修复

以下是一些错误码的具体修复方案

#### 少量存量，手动修复的错误码

对于一些存量较少的错误码，直接手动修复是效率最高且最安全的修复方式，这样的错误码主要包含：

- E713: test for membership should be ‘not in’
- E714: test for object identity should be ‘is not’
- F402: import `module` from line `N` shadowed by loop variable
- F522: `.format(...)` unused named arguments
- F524: `.format(...)` missing argument
- F541: f-string without any placeholders
- F601: dictionary key `name` repeated with different values
- F632: use `==`/`!=` to compare `str`, `bytes`, and `int` literals
- F631: assertion test is a tuple, which is always `True`
- W601: `.has_key()` is deprecated, use ‘in’
- W605: invalid escape sequence ‘x’
- 还有一些需要在 black 引入后再修复的，会在之后继续添加

#### 使用/编写工具，自动修复的错误码

对于一些存量不是特别大，但手动修复比较麻烦的错误码，建议使用现有的工具或者编写脚本来进行修复，这样的错误码主要包含：

- E711: comparison to None should be ‘if cond is None:’
- E712: comparison to True should be ‘if cond is True:’ or ‘if cond:’
- 可能还有些没考虑到的，需要之后进一步确定修复方案

#### F401（逐目录修复）

F401 为 import 了未使用的模块，该问题存量非常大，因此使用 autoflake 来进行自动修复

由于在[尝试全量一次性修复](https://github.com/PaddlePaddle/Paddle/pull/45252/)后发生了难以定位的错误，因此该错误码需要逐目录来做，优先修复单测和 tools 这种不会被其他模块 import 的目录，之后逐渐向较为核心的模块进行修复。

以 `python/paddle/fluid/tests/unittests/asp/` 为例，在 Paddle 根目录执行以下命令即可一次性清除该目录下全部 F401 问题

```bash
autoflake --in-place --remove-all-unused-imports --exclude=__init__.py --ignore-pass-after-docstring --recursive ./python/paddle/fluid/tests/unittests/asp/
```

如果清除后发现 CI 无法通过，需要根据情况判断问题，如果该 import 是必要的，应当在该 import 处添加 `# noqa: F401` 以 disable 该处报错。

## 三、任务分工和排期规划

由 Flake8 小组自行认领任务，每人负责部分错误码的部分目录。由于任务尚未开始，具体排期尚无法确定。

预计 2022 年底应该能完成绝大多数错误码的修复。

目前 Flake8 小组成员如下：

- [Nyakku Shigure](https://github.com/SigureMo)
- [Shuangchi He](https://github.com/Yulv-git)
- [Tony Cao](https://github.com/caolonghao)
- [Zheng_Bicheng](https://github.com/Zheng-Bicheng)
- [Infinity_lee](https://github.com/enkilee)

## 四、其他注意事项及风险评估

由于 fluid 预计将会在 Paddle 2.5 中移除，因此 flake8 忽略 `python/paddle/fluid/` 目录下的错误码，但其中的 `python/paddle/fluid/tests/` 目录仍需要进行修复。

## 五、影响面

开发人员在后续开发过程中需要遵守 Flake8 的规范，否则无法通过 pre-commit 和 CI，能够极大提高 Paddle Python 代码的质量。

## 名词解释

- Linter：代码检查工具（含代码风格、语法、逻辑等等）
- Formatter：代码格式化工具

## 附件及参考资料

- [Flake8 Error Code](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- [pycodestyle Error Code](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes)
- [Flake8 tracking issue](https://github.com/PaddlePaddle/Paddle/issues/46039)
- [Flake8 小组协作文档](https://cattidea.github.io/paddle-flake8-project/)